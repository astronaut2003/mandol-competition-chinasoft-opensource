"""V3.1 real SQLite/API integration; model doubles are explicitly unit-only."""
import json,sqlite3,time
from pathlib import Path
from dataclasses import replace
import httpx,pytest
from fastapi.testclient import TestClient
from mandol_competition.chinasoft.config import Settings
from mandol_competition.chinasoft.service import Platform
from mandol_competition.chinasoft.api import create_app
from mandol_competition.chinasoft.models import ModelError,RetrievalModels
from mandol_competition.chinasoft.conversation import recent_turns,attachment_context,plan_turn,generation_messages,citation_numbers
from mandol_competition.chinasoft.repository import parse_source
from mandol_competition.chinasoft.routing import routes
from mandol_competition.chinasoft.attachments import save_attachment
from mandol_competition.chinasoft.schemas import AddMemory
from tests.chinasoft.auth_helpers import login_test,api_owner

@pytest.fixture
def platform(tmp_path):
    p=Platform(Settings(data_dir=tmp_path,sparse_url='',allow_retrieval_degraded=True))
    yield p
    p.close()


def original(p,owner,w,s,content='我喜欢篮球。',files=None,request='raw'):
    r=p.store.accept(owner,AddMemory(workspace_id=w,session_id=s,request_id=request,messages=[{'content':content}],attachment_ids=files or []))
    job=p.store.claim('ingest','v31');assert job
    result=p.do_ingest(job);p.store.finish(job,result)
    return r


def test_chat_generates_without_waiting_for_memory_and_replays_complete_turns(platform,monkeypatch):
    p=platform;captured=[]
    async def model(messages,vision=False):
        captured.append(messages);yield '单元测试生成内容，不是真实模型验收。'
    monkeypatch.setattr(p.generation,'stream',model)
    monkeypatch.setattr(p.generation,'complete',lambda *a,**kw:{'content':json.dumps({'intent':'recent_context','retrieve':False,'query':''}),'response_model':'deepseek-flash'})
    monkeypatch.setattr(p,'search',lambda *a,**k:(_ for _ in ()).throw(AssertionError('unnecessary retrieval')))
    with TestClient(create_app(platform=p,start_workers=False)) as c:
        identity=login_test(c,p);w=identity['workspaces'][0]['id'];s=c.post('/api/sessions',json={'workspace_id':w}).json()['id']
        data={'workspace_id':w,'session_id':s,'request_id':'greet','content':'你好'}
        r=c.post('/api/chat',json=data);assert 'event: done' in r.text and 'event: error' not in r.text
        # Ingestion is intentionally still pending: generation did not wait or pretend index readiness.
        owner=api_owner(c,p);assert p.store.rows('SELECT status FROM cs_ingestions WHERE owner=?',(owner,))[0]['status']=='pending'
        r=c.post('/api/chat',json={**data,'request_id':'follow','content':'请继续上面的解释。'})
        assert 'event: done' in r.text
        messages=captured[-1];assert any(m['role']=='assistant' and '单元测试' in m['content'] for m in messages)
        assert not any('只能根据' in str(m['content']) for m in messages)
        assert not p.store.rows('SELECT 1 FROM memories')
        replay=c.post('/api/chat',json=data);assert '"cached":true' in replay.text and len(captured)==2
        assert len(c.get('/api/sessions',params={'workspace_id':w}).json()['items'])>=2


@pytest.mark.parametrize('name,code,symbol',[
 ('demo.py','def double(x):\n    return x * 2\n','double'),
 ('demo.ts','export function double(x: number) { return x * 2; }','double'),
 ('demo.tsx','export const View = () => <div>Hello</div>;','View'),
 ('demo.rs','pub fn double(x: i32) -> i32 { x * 2 }','double'),
 ('demo.go','package main\nfunc double(x int) int { return x * 2 }','double'),
 ('Demo.java','class Demo { int doubleValue(int x) { return x * 2; } }','doubleValue'),
 ('demo.cpp','int twice(int x) { return x * 2; }','twice')])
def test_real_tree_sitter_languages(name,code,symbol):
    parsed=parse_source(name,code)
    assert not parsed['error'] and parsed['parser']=='tree-sitter-1.14.3'
    assert any(s['name']==symbol for s in parsed['symbols'])
    assert all(c['resolved'] is False for c in parsed['calls'])


@pytest.mark.parametrize('name,data,mime,status',[
 ('../escape.py',b'print(1)','text/plain',415),
 ('bad.py',b'\0binary','text/plain',415),
 ('bad.py',b'print(1)','image/png',415),
 ('archive.zip',b'PK\x03\x04','application/zip',415),
 ('secret.py',b'api_key = "sk-example-abcdefghijklmnopqrstuvwxyz"','text/plain',422),
 ('large.py',b'x'*262145,'text/plain',413),
 ('bad.json',b'{broken','application/json',422),
 ('bad.toml',b'a = [','text/plain',422)])
def test_file_validation_is_durable_safe_and_isolated(platform,name,data,mime,status):
    p=platform;owner='a'*32;w=p.store.create_workspace(owner,'attachments')['id'];s=p.store.create_session(owner,w,'textual','test',unified=True)['id']
    with pytest.raises(Exception) as error:save_attachment(p,owner,w,s,name,data,mime)
    assert error.value.status==status and not p.store.rows('SELECT * FROM cs_attachments')


def test_attachment_version_invalidates_sources_and_direct_context_before_embedding(platform):
    p=platform;o='a'*32;w=p.store.create_workspace(o,'files')['id'];s=p.store.create_session(o,w,'textual','files',unified=True)['id']
    first=save_attachment(p,o,w,s,'utils.py',b'def old_name(x):\n    return x * 2\n','text/x-python')
    receipt=original(p,o,w,s,'解释这份代码',[first['id']]);scope=p.store.scope(o,w,'textual')
    raw=p.state.get_memories(scope,receipt['memory_ids'])[receipt['memory_ids'][-1]]
    assert raw.metadata['attachment_id']==first['id'] and p.eligible(scope,raw)
    pending=p.store.ingestion(o,receipt['ingestion_id']);job=p.store.rows("SELECT * FROM cs_jobs WHERE kind='derive' AND json_extract(payload,'$.route')='source_code'")[0]
    result=p.derive_attachment_code(scope,[raw],pending,job);assert result['generated']==1
    symbol=p.state.get_memories(scope,result['memory_ids'])[result['memory_ids'][0]];assert p.eligible(scope,symbol)
    blocks,public=attachment_context(p,o,w,s,[first['id']],[],'old_name')
    assert '1: def old_name' in blocks[0]['untrusted_content'] and public[0]['start_line']==1
    second=save_attachment(p,o,w,s,'utils.py',b'def new_name(x):\n    return x * 3\n','text/plain')
    assert second['version']==2 and second['id']!=first['id']
    assert not p.eligible(scope,raw) and not p.eligible(scope,symbol)
    assert p.state.get_memories(scope,[raw.logical_uid])[raw.logical_uid].text_content==raw.text_content
    assert attachment_context(p,o,w,s,[first['id']],[],'old_name')==([],[])
    with pytest.raises(Exception):p.store.attachment('b'*32,second['id'],w,s)
    same=save_attachment(p,o,w,s,'utils.py',b'def new_name(x):\n    return x * 3\n','text/plain');assert same['version']==2
    # Reactivating immutable old content gets a new attachment version, not old eligibility.
    reverted=save_attachment(p,o,w,s,'utils.py',b'def old_name(x):\n    return x * 2\n','text/plain')
    assert reverted['version']==3 and not p.eligible(scope,raw)


def test_attachment_api_immediate_context_followup_and_soft_delete(platform,monkeypatch):
    captured=[]
    async def model(messages,vision=False):captured.append(messages);yield '源码来自 `helper.py:1-2`，这是测试输出。'
    monkeypatch.setattr(platform.generation,'stream',model)
    with TestClient(create_app(platform=platform,start_workers=False)) as c:
        w=login_test(c,platform)['workspaces'][0]['id'];s=c.post('/api/sessions',json={'workspace_id':w}).json()['id'];fields={'workspace_id':w,'session_id':s}
        f=c.post('/api/attachments',data=fields,files={'file':('helper.py',b'def increment(x):\n    return x + 1\n','text/plain')}).json()
        data={**fields,'request_id':'file-chat','content':'解释这个文件','attachment_ids':[f['id']]}
        response=c.post('/api/chat',json=data);assert 'event: done' in response.text and 'event: error' not in response.text
        assert 'return x + 1' in str(captured[-1]) and 'source_locator' in str(captured[-1])
        c.post('/api/chat',json={**fields,'request_id':'file-follow','content':'把它处理零输入的过程展开说说'})
        assert 'return x + 1' in str(captured[-1])
        history=c.get('/api/chat/history',params=fields).json()['items'];assert history[0]['attachment_files'][0]['name']=='helper.py'
        assert c.delete('/api/attachments/'+f['id'],params={'workspace_id':w}).status_code==200
        history=c.get('/api/chat/history',params=fields).json()['items'];assert all(r['status']=='stale' for r in history)
        assert platform.store.rows('SELECT text FROM cs_attachments')[0]['text'].startswith('def increment')


def test_optional_planning_failure_is_bounded_and_identity_uses_runtime(platform,monkeypatch):
    calls=[]
    def complete(*args,**kwargs):calls.append(kwargs);raise ModelError('invalid_json','Injected malformed JSON')
    monkeypatch.setattr(platform.generation,'complete',complete)
    assert not plan_turn(platform,'你是什么模型？',[],[],'')['retrieve'] and not calls
    plan=plan_turn(platform,'上次处理请求幂等性问题的结论是什么？',[],[],'')
    assert plan['retrieve'] and len(calls)==1 and calls[0]['attempts']==1 and calls[0]['foreground']
    messages=generation_messages(platform,'你是什么模型',[],[],[],[])
    assert platform.settings.llm_request_model in messages[0]['content']
    assert citation_numbers('数组 `[1]` 和代码\n```py\nx=[2]\n```\n事实[3]')=={3}


def test_questions_and_assistant_files_do_not_become_profiles():
    assert routes({'role':'assistant','content':'我喜欢茶'})==[]
    assert routes({'role':'attachment','attachment_id':'a','attachment_kind':'code','content':'# 我喜欢茶'})==['source_code']
    assert routes({'role':'user','content':'你是什么模型？'})==[]
    assert routes({'role':'user','content':'你好，我叫小明，喜欢篮球。'})==['conversation']
    assert set(routes({'role':'user','content':'我昨天修复了数据库幂等问题，下周计划补并发测试。'}))=={'conversation','engineering'}


def test_projection_upgrade_preserves_semantic_profile_and_uses_distinct_frontier(tmp_path):
    old=Platform(Settings(data_dir=tmp_path,sparse_url='',allow_retrieval_degraded=True));o='a'*32;w=old.store.create_workspace(o,'old')['id'];s=old.store.create_session(o,w,'textual','old',unified=True)['id'];r=original(old,o,w,s);profile=old.profile.hash;rows=old.state._rows('SELECT * FROM memories');old.close()
    p=Platform(Settings(data_dir=tmp_path,sparse_url='http://127.0.0.1:19635'))
    try:
        assert p.profile.hash==profile and p.vector_profile.hash!=profile and p.projection_channel!='dense'
        assert p.state._rows('SELECT * FROM memories')==rows
        assert p.vector_collection!=p.settings.collection
        assert p.state.identity.collection==p.settings.collection
        with pytest.raises(RuntimeError):p.versioned_projection.state.prepare
        p.versioned_projection.schedule();jobs=p.store.rows("SELECT * FROM cs_jobs WHERE json_extract(payload,'$.projection_profile') IS NOT NULL")
        assert len(jobs)==1
        p.versioned_projection.last_schedule=0;p.versioned_projection.schedule()
        assert len(p.store.rows("SELECT * FROM cs_jobs WHERE json_extract(payload,'$.projection_profile') IS NOT NULL"))==1
    finally:p.close()


@pytest.mark.parametrize('backend,code',[('vllm','retrieval_required'),('legacy-http','splade_required')])
def test_missing_splade_default_fails_start_without_lying_about_health(tmp_path,backend,code):
    p=Platform(Settings(data_dir=tmp_path,retrieval_backend=backend,sparse_url='http://127.0.0.1:1',allow_retrieval_degraded=False))
    try:
        with pytest.raises(ModelError) as error:p.start()
        assert error.value.code==code
        assert not p.retrieval.probe_sparse()['ready']
    finally:p.close()


def test_sparse_adapter_validates_identity_query_mode_and_vocabulary(platform,monkeypatch):
    p=platform;s=p.settings;s.sparse_url='http://sparse.unit';requests=[]
    # This is the legacy adapter's wire contract, not native vLLM /pooling.
    # V4 keeps the legacy regression explicit after the default backend changed.
    p.retrieval.close();p.retrieval=RetrievalModels(s,p.processor)
    def handler(req):
        requests.append(req)
        if req.method=='GET':return httpx.Response(200,json={'ready':True,'model':RetrievalModels.SPLADE_MODEL,'revision':RetrievalModels.SPLADE_REVISION,'pooling':RetrievalModels.SPLADE_POOLING,'vocabulary_size':30522,'max_length':512,'query_mode':'full','document_mode':'full','weights_sha256':RetrievalModels.SPLADE_WEIGHTS,'tokenizer_sha256':RetrievalModels.SPLADE_TOKENIZER,'config_sha256':RetrievalModels.SPLADE_CONFIG})
        mode=json.loads(req.content)['mode']
        return httpx.Response(200,json={'model':RetrievalModels.SPLADE_MODEL,'revision':RetrievalModels.SPLADE_REVISION,'pooling':RetrievalModels.SPLADE_POOLING,'mode':mode,'data':[{'index':0,'sparse':{'30522':1.0}}]})
    p.retrieval.http.close();p.retrieval.http=httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(ModelError):p.retrieval.sparse('database')
    assert json.loads(requests[-1].content)['mode']=='query' and not p.retrieval.sparse_health['ready']


def test_current_question_and_its_descendants_are_not_historical_evidence(platform):
    from mandol_substrate.state import CanonicalMemory,Mutation
    from mandol_substrate.state.models import digest
    p=platform;o='a'*32;w=p.store.create_workspace(o,'current')['id'];s=p.store.create_session(o,w,'textual','current',unified=True)['id']
    receipt=original(p,o,w,s,'我之前说过喜欢什么运动？');scope=p.store.scope(o,w,'textual');source=receipt['memory_ids'][0]
    derived=CanonicalMemory('question-summary','用户询问之前喜欢什么运动',metadata={'source_versions':{source:1}},session_id=s,memory_type='summary',memory_level='L1_Summary')
    p.coordinator.apply(scope,'question-summary','question-summary',Mutation(memories=(derived,)),{})
    assert p.search(o,w,'喜欢什么运动',session_id=s,conversation_first=True)['items']
    assert p.search(o,w,'喜欢什么运动',session_id=s,conversation_first=True,exclude_sources=[source])['items']==[]


def test_repeated_file_derivation_reuses_unmodified_symbols(platform):
    p=platform;o='a'*32;w=p.store.create_workspace(o,'parse')['id'];s=p.store.create_session(o,w,'textual','parse',unified=True)['id']
    f=save_attachment(p,o,w,s,'keep.py',b'def keep(x):\n    return x\n')
    receipt=original(p,o,w,s,files=[f['id']]);scope=p.store.scope(o,w,'textual');ing=p.store.ingestion(o,receipt['ingestion_id']);sources,_=p.session_sources(scope,s)
    first=p.derive_attachment_code(scope,sources,ing,{'id':'parse-one','owner':o})
    seq=p.state.scope_state(scope)['published_seq']
    second=p.derive_attachment_code(scope,sources,ing,{'id':'parse-two','owner':o})
    assert first['generated']==1 and second['generated']==0 and second['reused']==1
    assert p.state.scope_state(scope)['published_seq']==seq


def test_projection_fault_classification_uses_legal_versioned_channel(tmp_path):
    p=Platform(Settings(data_dir=tmp_path,sparse_url='http://127.0.0.1:1',allow_retrieval_degraded=False))
    try:
        o='a'*32;w=p.store.create_workspace(o,'fault')['id'];s=p.store.create_session(o,w,'textual','fault',unified=True)['id'];original(p,o,w,s)
        scope=p.store.scope(o,w,'textual')
        p.versioned_projection.state.mark_projection_degraded(scope,'dense','InjectedFault')
        rows=p.state._rows('SELECT * FROM projection_frontiers WHERE scope=? AND channel=?',(scope.key,p.projection_channel))
        assert rows[0]['status']=='degraded'
    finally:p.close()


def test_engineering_source_hash_uses_the_builders_actual_provenance_contract(platform):
    from mandol_substrate.state import CanonicalMemory,Mutation
    from mandol_substrate.representation import MemoryUnit
    from mandol_competition.code.software.episodes import source_digest
    import hashlib
    p=platform;o='a'*32;w=p.store.create_workspace(o,'engineering-hash')['id'];s=p.store.create_session(o,w,'textual','engineering',unified=True)['id']
    r=original(p,o,w,s,'我修复了请求幂等边界。');scope=p.store.scope(o,w,'textual');source=p.state.get_memories(scope,r['memory_ids'])[r['memory_ids'][0]]
    unit=MemoryUnit(uid=source.logical_uid,raw_data={'text_content':source.text_content},metadata={'user_id':scope.user_id,'session_id':s,'request_id':source.request_id,'role':source.role,'ts_ms':source.business_timestamp_ms,'sequence':source.metadata['sequence']})
    correct=source_digest(unit);metadata={'source_versions':{source.logical_uid:1},'source_hashes':{source.logical_uid:correct},'memory_family':'code'}
    m=CanonicalMemory('engineering-digest-test','明确标注的测试经验',metadata=metadata,session_id=s,memory_type='engineering_experience',memory_level='L1_Episode')
    p.coordinator.apply(scope,'engineering-digest-test','hash-test',Mutation(memories=(m,)),{})
    current=p.state.get_memories(scope,[m.logical_uid])[m.logical_uid];assert p.eligible(scope,current)
    wrong=replace(current,metadata={**metadata,'source_hashes':{source.logical_uid:hashlib.sha256(source.text_content.encode()).hexdigest()}})
    assert not p.eligible(scope,wrong)
    p.coordinator.apply(scope,'changed-input','changed',Mutation(memories=(replace(source,version=2,text_content='修订后的来源'),)),{})
    assert not p.eligible(scope,current)


def test_attachment_planning_can_leave_unrelated_files_out(platform,monkeypatch):
    files=[{'source_locator':{'path':'calc.py'},'untrusted_content':'1: def triple(x): return x * 3'}]
    monkeypatch.setattr(platform.generation,'complete',lambda *a,**kw:{'content':json.dumps({'intent':'general','retrieve':False,'use_attachments':False,'query':''}),'response_model':'deepseek-flash'})
    assert not plan_turn(platform,'什么是 HTTP 协议？',[],files,'')['use_attachments']
    assert plan_turn(platform,'解释本次文件',[],files,'',True)['use_attachments']
    assert not plan_turn(platform,'你是什么模型？',[],files,'')['use_attachments']
    assert not plan_turn(platform,'读出本轮图片的文字',[],files,'',False,True)['use_attachments']
    assert routes({'role':'user','content':'请描述这张图片直接可见的内容。','media_ids':['image']})==['visual']


def test_json_credentials_rejected_but_environment_lookup_is_code(platform):
    p=platform;o='a'*32;w=p.store.create_workspace(o,'sensitive')['id'];s=p.store.create_session(o,w,'textual','files',unified=True)['id']
    with pytest.raises(Exception) as error:save_attachment(p,o,w,s,'config.json',b'{"password":"actual-looking-example-value"}','application/json')
    assert error.value.code=='sensitive_attachment'
    saved=save_attachment(p,o,w,s,'config.py',b'import os\npassword = os.getenv("DATABASE_PASSWORD")\n','text/x-python')
    assert saved['id']
