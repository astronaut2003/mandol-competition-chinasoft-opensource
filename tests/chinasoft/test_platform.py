from tests.chinasoft.auth_helpers import login_test,api_owner
"""SQLite/API integration and deterministic fault injection (not live model QA).

Live inference/demo/browser reports are produced separately under docs/verification.
"""
from pathlib import Path
from dataclasses import replace
import io,json,time
import httpx,pytest
from fastapi.testclient import TestClient
from mandol_substrate.state import CanonicalMemory,Mutation
from mandol_substrate.state.models import digest
from mandol_competition.chinasoft.api import create_app
from mandol_competition.chinasoft.config import Settings
from mandol_competition.chinasoft.service import Platform
from mandol_competition.chinasoft.schemas import AddMemory
from mandol_competition.chinasoft.models import ModelError


@pytest.fixture
def platform(tmp_path,monkeypatch):
    monkeypatch.setenv('UNIT_MODEL_KEY','unit-key')
    p=Platform(Settings(data_dir=tmp_path,llm_api_key_env='UNIT_MODEL_KEY'))
    yield p
    p.close()


def setup(p):
    owner='a'*32;w=p.store.create_workspace(owner,'test');s=p.store.create_session(owner,w['id'],'code','test')
    return owner,w['id'],s['id']


def request(w,s,id='r',content='SQLite保存原始事实'):
    return AddMemory(workspace_id=w,session_id=s,request_id=id,mode='code',messages=[{'role':'user','content':content}])


def ingest(p):
    job=p.store.claim('ingest','test-owner');assert job
    result=p.do_ingest(job);p.store.finish(job,result);return job,result


def test_durable_acceptance_before_workers_and_restart(tmp_path,monkeypatch):
    p=Platform(Settings(data_dir=tmp_path));owner,w,s=setup(p);r=p.store.accept(owner,request(w,s))
    assert p.store.ingestion(owner,r['ingestion_id'])['status']=='pending'
    assert not p.explorer(owner,w)['items']
    claimed=p.store.claim('ingest','crashed-worker');assert claimed['status']=='pending';p.close()
    p=Platform(Settings(data_dir=tmp_path));p.store.recover();job,result=ingest(p)
    assert p.store.ingestion(owner,r['ingestion_id'])['status']=='saved'
    assert p.explorer(owner,w)['items'][0]['text']=='SQLite保存原始事实'
    assert job['attempts']==2
    assert p.store.accept(owner,request(w,s))['memory_ids']==r['memory_ids']
    assert p.explorer(owner,w)['total']==1;p.close()


def test_queue_backpressure_and_replay(platform):
    p=platform;p.settings.queue_capacity=3;owner,w,s=setup(p);r=p.store.accept(owner,request(w,s))
    assert p.store.accept(owner,request(w,s))==r
    with pytest.raises(Exception) as full:p.store.accept(owner,request(w,s,'r2'))
    assert full.value.code=='queue_full';assert len(p.store.jobs(owner,w)['items'])==3


def test_id_conflict_does_not_create_jobs(platform):
    owner,w,s=setup(platform);platform.store.accept(owner,request(w,s))
    with pytest.raises(Exception):platform.store.accept(owner,request(w,s,content='different'))
    assert len(platform.store.jobs(owner,w)['items'])==3


def test_source_version_invalidates_derived_without_erasing_l0(platform):
    p=platform;owner,w,s=setup(p);r=p.store.accept(owner,request(w,s));ingest(p);scope=p.store.scope(owner,w,'code')
    raw=p.state.get_memories(scope,r['memory_ids'])[r['memory_ids'][0]]
    derived=CanonicalMemory('derived','保存经验',metadata={'source_versions':{raw.logical_uid:1}},memory_type='engineering_experience',memory_level='L1_Episode')
    p.coordinator.apply(scope,'derived','d',Mutation(memories=(derived,)),{})
    assert p.eligible(scope,derived)
    updated=replace(raw,version=2,text_content='修订后的事实')
    p.coordinator.apply(scope,'revision','v2',Mutation(memories=(updated,)),{})
    assert not p.eligible(scope,derived)
    assert p.state.get_memories(scope,[raw.logical_uid])[raw.logical_uid].text_content=='修订后的事实'
    p.coordinator.apply(scope,'delete','delete',Mutation(delete_uids=(raw.logical_uid,)),{})
    assert not p.eligible(scope,derived)


def test_api_auth_scope_and_schema_validation(platform):
    app=create_app(platform=platform,start_workers=False)
    with TestClient(app) as a:
        auth=login_test(a,platform);w=auth['workspaces'][0]['id'];s=a.get('/api/sessions',params={'workspace_id':w}).json()['items'][0]['id']
        data={'workspace_id':w,'session_id':s,'request_id':'api','mode':'textual','messages':[{'content':'长期记忆测试','role':'user'}]}
        assert a.post('/api/memories',json=data).status_code==202
        assert a.post('/api/memories',json=data).status_code==202
        assert a.post('/api/memories',json={**data,'messages':[{'content':'bad','role':'system'}]}).status_code==422
        assert a.post('/api/memories',json=data,headers={'Origin':'https://evil.invalid'}).status_code==403
        cookies=dict(a.cookies);a.cookies.clear();login_test(a,platform)
        assert a.get('/api/memories',params={'workspace_id':w}).status_code==404
        a.cookies.clear();a.cookies.update(cookies);ingest(platform)
        mem=a.get('/api/memories',params={'workspace_id':w}).json()['items'][0]
        assert a.get('/api/memories/'+mem['id'],params={'workspace_id':w}).json()['text']=='长期记忆测试'


def test_image_real_bytes_and_session_ownership(platform):
    app=create_app(platform=platform,start_workers=False)
    image=(Path(__file__).resolve().parents[2]/'demo/status_board.png').read_bytes()
    with TestClient(app) as c:
        w=login_test(c,platform)['workspaces'][0]['id'];s=c.post('/api/sessions',json={'workspace_id':w,'mode':'multimodal','title':'visual'}).json()['id']
        fields={'workspace_id':w,'session_id':s};r=c.post('/api/media',data=fields,files={'file':('board.png',image,'image/png')})
        assert r.status_code==201;mid=r.json()['id'];assert c.get('/api/media/'+mid,params={'workspace_id':w}).content==image
        thumb=c.get('/api/media/'+mid,params={'thumb':True,'workspace_id':w});assert thumb.status_code==200;assert len(thumb.content)<len(image)
        assert c.post('/api/media',data=fields,files={'file':('bad.png',b'not-an-image','image/png')}).status_code==415
        other=c.post('/api/sessions',json={'workspace_id':w,'mode':'multimodal','title':'other'}).json()['id']
        bad={'workspace_id':w,'session_id':other,'mode':'multimodal','request_id':'cross-session','messages':[{'content':'test','role':'user','media_ids':[mid]}]}
        assert c.post('/api/memories',json=bad).status_code==404
        c.cookies.clear();login_test(c,platform);assert c.get('/api/media/'+mid,params={'workspace_id':w}).status_code==404


def mock_response(content='{"ok":true}',model='deepseek-flash',finish='stop'):
    return {'model':model,'choices':[{'finish_reason':finish,'message':{'content':content}}],'usage':{}}


def configure(p,handler):
    p.generation.client.close();p.generation.client=httpx.Client(transport=httpx.MockTransport(handler),timeout=1)
    p.generation.retry_sleep=lambda delay:None


@pytest.mark.parametrize('status',[429,500,503])
def test_bounded_http_retry_and_failure(platform,status):
    calls=[]
    def handler(req):calls.append(req);return httpx.Response(status,json={'error':'injected'})
    configure(platform,handler)
    with pytest.raises(ModelError) as failure:platform.generation.complete([{'role':'user','content':'test'}])
    assert len(calls)==3 and failure.value.retryable
    assert platform.store.rows('SELECT state FROM cs_calls')[0]['state']=='failed'


def test_429_then_success_has_durable_receipt(platform):
    calls=[]
    def handler(req):calls.append(1);return httpx.Response(429 if len(calls)<2 else 200,json={} if len(calls)<2 else mock_response())
    configure(platform,handler);messages=[{'role':'user','content':'JSON'}]
    first=platform.generation.complete(messages);second=platform.generation.complete(messages)
    assert first==second and len(calls)==2


@pytest.mark.parametrize('content,model,finish,code',[
    ('','deepseek-flash','stop','empty_response'),
    ('{"ok":','deepseek-flash','stop','invalid_json'),
    ('{"ok":true,"ok":false}','deepseek-flash','stop','invalid_json'),
    ('{"ok":"yes"}','deepseek-flash','stop','invalid_json'),
    ('{"ok":true}','other-model','stop','model_identity'),
    ('{"ok":true}','deepseek-flash','length','incomplete_response'),
    ('{"ok":NaN}','deepseek-flash','stop','invalid_json')])
def test_structured_rejections_do_not_create_memory(platform,content,model,finish,code):
    configure(platform,lambda req:httpx.Response(200,json=mock_response(content,model,finish)))
    schema={'type':'object','properties':{'ok':{'type':'boolean'}},'required':['ok'],'additionalProperties':False}
    with pytest.raises(ModelError) as error:platform.generation.complete([{'role':'user','content':'JSON test'}],schema=schema)
    assert error.value.code==code
    assert not platform.state._rows('SELECT * FROM memories')
    assert platform.store.rows('SELECT state FROM cs_calls')[0]['state']=='failed'


def test_unknown_timeout_is_not_silently_repeated_after_restart(tmp_path,monkeypatch):
    monkeypatch.setenv('UNIT_MODEL_KEY','unit-key');s=Settings(data_dir=tmp_path,llm_api_key_env='UNIT_MODEL_KEY');p=Platform(s);calls=[]
    def handler(req):calls.append(1);raise httpx.ReadTimeout('injected',request=req)
    configure(p,handler);messages=[{'role':'user','content':'unknown outcome'}]
    with pytest.raises(ModelError) as error:p.generation.complete(messages)
    assert error.value.unknown and len(calls)==1;p.close();p=Platform(s);configure(p,handler)
    with pytest.raises(ModelError) as error:p.generation.complete(messages)
    assert error.value.code=='unknown_remote' and len(calls)==1;p.close()


def test_model_and_recipe_fingerprints_do_not_reuse_old_cache(platform):
    from mandol_competition.textual.core.hlm_production import ProductionHLMRecipe
    old=ProductionHLMRecipe(platform.profile.hash)
    assert old.fingerprint!=platform.recipe.fingerprint
    assert platform.recipe.profile_recipe.model=='deepseek-flash'
    from mandol_competition.code.experience import ExperienceBuilder,FINGERPRINT
    from types import SimpleNamespace
    b=ExperienceBuilder(SimpleNamespace(store=SimpleNamespace(state=platform.state)),generator=lambda *a:None,model_plan=platform.settings.contract())
    assert b.fingerprint!=FINGERPRINT


def test_l0_survives_vlm_and_qdrant_failure(platform,monkeypatch):
    p=platform;owner,w,s=setup(p);p.store.accept(owner,request(w,s));ingest(p)
    index=p.store.claim('index','test-owner');monkeypatch.setattr(p.coordinator,'project_optional',lambda scope:False)
    with pytest.raises(ModelError) as error:p.do_index(index)
    p.store.fail(index,error.value)
    assert p.store.jobs(owner,w)['items'][0]['status'] in ('pending','retry_wait','completed')
    assert p.explorer(owner,w)['total']==1 and p.explorer(owner,w)['items'][0]['lexical_ready']
    configure(p,lambda req:httpx.Response(503,json={'error':'vision unavailable'}))
    with pytest.raises(ModelError):p.generation.complete([{'role':'user','content':'image'}],vision=True)
    assert p.explorer(owner,w)['total']==1


def test_chat_without_key_reports_real_error_and_preserves_question(platform,monkeypatch):
    monkeypatch.delenv('UNIT_MODEL_KEY');app=create_app(platform=platform,start_workers=False)
    with TestClient(app) as c:
        w=login_test(c,platform)['workspaces'][0]['id'];s=c.get('/api/sessions',params={'workspace_id':w}).json()['items'][0]['id']
        r=c.post('/api/chat',json={'workspace_id':w,'session_id':s,'request_id':'no-key','content':'过去的决策是什么？','mode':'auto','media_ids':[]})
        assert 'event: accepted' in r.text and 'model_not_configured' in r.text
        ingest(platform);assert c.get('/api/memories',params={'workspace_id':w}).json()['total']==1
        history=c.get('/api/chat/history',params={'workspace_id':w,'session_id':s}).json()['items']
        assert history[0]['status']=='failed';assert 'fake' not in history[0]['answer']


def test_demo_reset_cannot_target_regular_workspace(platform):
    app=create_app(platform=platform,start_workers=False)
    with TestClient(app) as c:
        w=login_test(c,platform)['workspaces'][0]['id']
        assert c.delete('/api/demo/'+w).status_code==403


def test_lease_completion_cannot_be_stolen(platform):
    p=platform;owner,w,s=setup(p);p.store.accept(owner,request(w,s));job=p.store.claim('ingest','one')
    assert p.store.claim('ingest','two') is None
    p.store.finish(dict(job,lease='two'),{'stage':'invalid'})
    assert p.store.rows('SELECT status FROM cs_jobs WHERE id=?',(job['id'],))[0]['status']=='running'


def test_configuration_refuses_aml_resources():
    with pytest.raises(ValueError):Settings(collection='aml_memories')
    with pytest.raises(ValueError):Settings(port=18600)


def test_configured_code_model_identity_cannot_be_forged(platform):
    from types import SimpleNamespace
    from mandol_competition.code.experience import ExperienceBuilder
    p=platform;owner,w,s=setup(p)
    p.store.accept(owner,AddMemory(workspace_id=w,session_id=s,mode='code',request_id='long-code',messages=[
        {'role':'user','content':'JWT issuer validation failed in staging. '*6},
        {'role':'assistant','content':'Update issuer configuration and preserve source evidence. '*6}]))
    ingest(p);scope=p.store.scope(owner,w,'code');sources=list(p.state.get_memories(scope,[m['id'] for m in p.explorer(owner,w)['items']]).values())
    b=ExperienceBuilder(SimpleNamespace(store=SimpleNamespace(state=p.state,scope=lambda _:scope)),
        model_plan=p.settings.contract(),generator=lambda *args:({'task_summary':None,'diagnosis':[],'resolution':[],'knowledge':[]},{'response_model':'wrong-model'}))
    mutation=b.prepare(scope,s,'wrong-model-job','digest',sources)
    assert not mutation.memories
    assert p.explorer(owner,w)['total']==2


def test_concurrent_identical_media_upload_is_atomic(platform):
    from concurrent.futures import ThreadPoolExecutor
    p=platform;owner,w,_=setup(p);s=p.store.create_session(owner,w,'multimodal','images')['id']
    image=(Path(__file__).resolve().parents[2]/'demo/status_board.png').read_bytes()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(lambda _:p.save_media(owner,w,s,image),range(2)))
    assert results[0]['id']==results[1]['id']
    assert (p.settings.data_dir/results[0]['path']).read_bytes()==image
    assert len(p.store.rows('SELECT * FROM cs_media'))==1


def test_visual_append_uses_only_new_source_and_preserves_window(platform):
    from PIL import Image
    p=platform;owner,w,_=setup(p);s=p.store.create_session(owner,w,'multimodal','images')['id']
    data=(Path(__file__).resolve().parents[2]/'demo/status_board.png').read_bytes()
    buffer=io.BytesIO();Image.open(io.BytesIO(data)).save(buffer,format='JPEG',quality=88)
    responses=[]
    def handler(req):
        body=json.loads(req.content);assert any(b['type']=='image_url' and b['image_url']['url'].startswith('data:image/') for b in body['messages'][-1]['content'])
        responses.append(1)
        return httpx.Response(200,json=mock_response(json.dumps({'description':'设备状态看板','visible_text':['ONLINE'],'objects':['board'],'uncertainties':[]})))
    configure(p,handler)
    for index,image in enumerate([data,buffer.getvalue()]):
        media=p.save_media(owner,w,s,image)
        p.store.accept(owner,AddMemory(workspace_id=w,session_id=s,request_id=f'image-{index}',mode='multimodal',messages=[{'role':'user','content':f'图片说明 {index}','media_ids':[media['id']]}]))
        ingest(p);job=p.store.claim('derive','test-owner');p.do_derive(job);p.store.finish(job,{'generated':1})
    assert len(responses)==2  # first image is not regenerated on the second Add.
    items=p.explorer(owner,w)['items'];assert sum(m['type']=='visual_memory' for m in items)==2
    window=next(m for m in items if m['type']=='visual_episode');assert window['eligible']
    scope,record,_=p.locate(owner,w,window['id'])
    assert len(record.metadata['source_refs'])==2
    assert all(p.state.get_memories(scope,[r['uid']])[r['uid']].memory_level=='L0_Observation' for r in record.metadata['source_refs'])


def test_unknown_code_build_can_retry_only_after_explicit_confirmation(platform):
    p=platform;owner,w,s=setup(p)
    p.store.accept(owner,AddMemory(workspace_id=w,session_id=s,mode='code',request_id='code-timeout',messages=[
        {'role':'user','content':'JWT issuer configuration fails in staging. '*6},
        {'role':'assistant','content':'Check the request ledger and source versions before retry. '*6}]))
    ingest(p);job=p.store.claim('derive','worker');p.generation.local.job_id=job['id'];p.generation.active_build_job=job['id']
    configure(p,lambda req:(_ for _ in ()).throw(httpx.ReadTimeout('injected',request=req)))
    with pytest.raises(ModelError) as failed:p.do_derive(job)
    p.store.fail(job,failed.value)
    assert failed.value.unknown
    with pytest.raises(Exception) as no_confirmation:p.store.retry(owner,job['id'])
    assert no_confirmation.value.code=='unknown_remote'
    p.store.retry(owner,job['id'],True)
    payload={'task_summary':None,'diagnosis':[],'resolution':[],'knowledge':[]}
    configure(p,lambda req:httpx.Response(200,json=mock_response(json.dumps(payload))))
    retry=p.store.claim('derive','worker');result=p.do_derive(retry);p.store.finish(retry,result)
    assert result['generated']==0  # Realistic empty valid extraction is not a fake experience.
    assert p.store.rows('SELECT status FROM cs_jobs WHERE id=?',(job['id'],))[0]['status']=='completed'
    assert p.explorer(owner,w)['total']==2


def test_historical_answer_is_marked_stale_after_source_revision(platform):
    from mandol_competition.chinasoft.store import now
    p=platform;app=create_app(platform=p,start_workers=False)
    with TestClient(app) as c:
        w=login_test(c,platform)['workspaces'][0]['id'];s=c.get('/api/sessions',params={'workspace_id':w}).json()['items'][0]['id']
        response=c.post('/api/memories',json={'workspace_id':w,'session_id':s,'request_id':'stale-source','mode':'textual','messages':[{'role':'user','content':'端口是20600'}]}).json()
        ingest(p);owner=api_owner(c,p);scope,raw,_=p.locate(owner,w,response['memory_ids'][0]);citation=p.serialize(scope,raw,detail=True)
        with p.state.transaction() as db:
            db.execute("INSERT INTO cs_chat(id,owner,workspace,session,request_id,digest,question,media,answer,citations,status,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",('historic',owner,w,s,'old-chat','digest','什么端口','[]','端口是20600。[1]',json.dumps([citation]),'completed',now(),now()))
        assert c.get('/api/chat/history',params={'workspace_id':w,'session_id':s}).json()['items'][0]['status']=='completed'
        p.coordinator.apply(scope,'source-revision','v2',Mutation(memories=(replace(raw,version=2,text_content='端口修订为19610'),)),{})
        assert c.get('/api/chat/history',params={'workspace_id':w,'session_id':s}).json()['items'][0]['status']=='stale'


def test_demo_reset_cancels_pending_and_removes_only_its_media(platform):
    p=platform;app=create_app(platform=p,start_workers=False)
    image=(Path(__file__).resolve().parents[2]/'demo/status_board.png').read_bytes()
    with TestClient(app) as c:
        regular=login_test(c,platform)['workspaces'][0]['id'];user=api_owner(c,p)
        demo=p.store.create_workspace(user,'demo',True);session=p.store.create_session(user,demo['id'],'multimodal','image')['id']
        media=p.save_media(user,demo['id'],session,image)
        receipt=p.store.accept(user,AddMemory(workspace_id=demo['id'],session_id=session,request_id='reset-pending',mode='multimodal',messages=[{'role':'user','content':'pending','media_ids':[media['id']]}]))
        assert c.delete('/api/demo/'+demo['id']).status_code==200
        assert c.get('/api/media/'+media['id'],params={'workspace_id':demo['id']}).status_code==404
        assert not (p.settings.data_dir/media['path']).exists()
        assert p.store.workspace(user,regular)['id']==regular
        assert all(r['status']=='cancelled' for r in p.store.rows('SELECT status FROM cs_jobs WHERE ingestion=?',(receipt['ingestion_id'],)))
