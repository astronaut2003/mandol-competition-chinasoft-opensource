"""V3 acceptance against real SQLite/API; provider faults explicitly injected."""
import io,json,sqlite3,time
from dataclasses import replace
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from mandol_substrate.state import CanonicalMemory,Mutation,Edge
from mandol_competition.chinasoft.api import create_app
from mandol_competition.chinasoft.config import Settings
from mandol_competition.chinasoft.service import Platform
from mandol_competition.chinasoft.auth import COOKIE,hashed,HASHER
from mandol_competition.chinasoft.schemas import AddMemory
from mandol_competition.chinasoft.visualization import MemoryViews

@pytest.fixture
def env(tmp_path,monkeypatch):
    monkeypatch.delenv('DEEPSEEK_API_KEY',raising=False);monkeypatch.delenv('SILICONFLOW_API_KEY',raising=False)
    p=Platform(Settings(data_dir=tmp_path,registration_enabled=True,registration_approval=False))
    with TestClient(create_app(platform=p,start_workers=False)) as c:
        c.headers['X-Mandol-Client']='studio'
        yield p,c

def register(c,name='user-alpha'):
    r=c.post('/api/auth/register',json={'username':name,'password':'correct-password-123'});assert r.status_code==201,r.text
    return login(c,name)

def login(c,name='user-alpha'):
    c.cookies.clear();r=c.post('/api/auth/login',json={'username':name,'password':'correct-password-123'});assert r.status_code==200,r.text
    d=r.json();c.headers['X-CSRF-Token']=d['csrf_token'];return d

def session(c,w,title='新会话'):
    r=c.post('/api/sessions',json={'workspace_id':w,'title':title});assert r.status_code==201,r.text;return r.json()['id']

def add(c,w,s,request='r',text='我喜欢清茶，昨天已完成周会；下周计划再去杭州。',**extra):
    r=c.post('/api/memories',json={'workspace_id':w,'session_id':s,'request_id':request,'messages':[{'content':text}],**extra});assert r.status_code==202,r.text;return r.json()

def ingest(p):
    j=p.store.claim('ingest','test');assert j;result=p.do_ingest(j);p.store.finish(j,result);return result


def test_registration_hash_cookie_csrf_and_revocation(env):
    p,c=env;identity=register(c);account=p.store.rows('SELECT * FROM cs_accounts')[0]
    assert account['role']=='user' and account['password_hash'].startswith('$argon2id$');assert HASHER.verify(account['password_hash'],'correct-password-123')
    cookie=c.cookies[COOKIE];record=p.store.rows('SELECT * FROM cs_auth_sessions')[0];assert record['token_hash']==hashed(cookie) and cookie not in str(record)
    assert c.post('/api/workspaces',json={'name':'bad'},headers={'X-CSRF-Token':''}).status_code==403
    assert c.post('/api/workspaces',json={'name':'bad'},headers={'Origin':'https://evil.invalid'}).status_code==403
    assert c.post('/api/auth/logout',json={}).status_code==200
    c.cookies.set(COOKIE,cookie);assert c.get('/api/auth/session').status_code==401


def test_login_limits_no_enumeration_and_expiry(env):
    p,c=env;register(c);c.cookies.clear()
    a=c.post('/api/auth/login',json={'username':'user-alpha','password':'wrong'})
    b=c.post('/api/auth/login',json={'username':'nonexistent','password':'wrong'})
    assert a.status_code==b.status_code==401 and a.json()==b.json()
    identity=login(c)
    with p.state.transaction() as db:db.execute('UPDATE cs_auth_sessions SET expires_at=0')
    assert c.get('/api/workspaces').status_code==401
    for _ in range(12):r=c.post('/api/auth/login',json={'username':'nonexistent','password':'wrong'})
    assert r.status_code==429


def test_registration_disabled_and_pending_no_auto_admin(env):
    p,c=env;p.settings.registration_enabled=False
    assert c.post('/api/auth/register',json={'username':'blocked','password':'correct-password-123'}).status_code==403
    p.settings.registration_enabled=True;p.settings.registration_approval=True
    r=c.post('/api/auth/register',json={'username':'pending','password':'correct-password-123'})
    assert r.status_code==201 and r.json()['status']=='pending' and r.json()['role']=='user'
    assert c.post('/api/auth/login',json={'username':'pending','password':'correct-password-123'}).status_code==401


def test_legacy_cookie_cannot_be_used_as_identity(env):
    p,c=env;o='f'*32;p.store.create_workspace(o,'private legacy');c.cookies.set('chinasoft_session',p.store.token(o))
    assert c.get('/api/workspaces').status_code==401
    identity=register(c);assert 'private legacy' not in str(identity)


def test_session_history_titles_archive_delete_and_stable_ingestion(env):
    p,c=env;w=register(c)['workspaces'][0]['id'];ids=[session(c,w) for _ in range(3)]
    for i,s in enumerate(ids):add(c,w,s,str(i),'背景 '+str(i));ingest(p)
    fourth=session(c,w);rows=c.get('/api/sessions',params={'workspace_id':w}).json()['items'];assert len(rows)==4 and {x['id'] for x in rows}==set(ids+[fourth])
    s=ids[0];r=add(c,w,s,'repeat','标题不会覆盖原有标题');assert add(c,w,s,'repeat','标题不会覆盖原有标题')==r
    changed=c.patch('/api/sessions/'+s,params={'workspace_id':w},json={'title':'手动标题'});assert changed.status_code==200
    add(c,w,s,'more','新的消息不应覆盖手动标题')
    assert c.get('/api/sessions/'+s,params={'workspace_id':w}).json()['title']=='手动标题'
    assert c.delete('/api/sessions/'+s,params={'workspace_id':w}).status_code==409
    c.patch('/api/sessions/'+s,params={'workspace_id':w},json={'archived':True})
    assert s not in str(c.get('/api/sessions',params={'workspace_id':w}).json()['items'])
    assert s in str(c.get('/api/sessions',params={'workspace_id':w,'archived':True}).json()['items'])
    c.delete('/api/sessions/'+s,params={'workspace_id':w,'confirm':True})
    assert c.get('/api/chat/history',params={'workspace_id':w,'session_id':s}).status_code==404
    assert len(c.get('/api/sessions',params={'workspace_id':w}).json()['items'])==3


@pytest.mark.parametrize('resource',['session','memory','relations','media','attachment','repository','files','map','jobs','events','graph','timeline','search','status','demo-reset','chat','repo-register','ingestion','job-retry'])
def test_cross_tenant_all_entry_points(env,resource):
    p,c=env;a=register(c);w=a['workspaces'][0]['id'];s=session(c,w);r=add(c,w,s);ingest(p);mid=r['memory_ids'][0]
    image=(Path(__file__).parents[2]/'demo/status_board.png').read_bytes()
    media=c.post('/api/media',data={'workspace_id':w,'session_id':s},files={'file':('board.png',image,'image/png')}).json()['id']
    att=c.post('/api/attachments',data={'workspace_id':w,'session_id':s},files={'file':('note.md',b'# evidence','text/plain')}).json()['id']
    register(c,'user-beta')
    paths={'session':'/sessions/'+s,'memory':'/memories/'+mid,'relations':'/memories/'+mid+'/relations','media':'/media/'+media,'attachment':'/attachments/'+att,'repository':'/repositories/guess','files':'/repositories/guess/files','map':'/repositories/guess/map','jobs':'/jobs','events':'/events','graph':'/memory-graph','timeline':'/sessions/'+s+'/timeline','search':'/memories/search','status':'/status','ingestion':'/ingestions/'+r['ingestion_id'],'job-retry':'/jobs/'+r['jobs'][0]+'/retry'}
    params={'workspace_id':w,'session_id':s,'q':'清茶'}
    if resource=='demo-reset':resp=c.delete('/api/demo/'+w)
    elif resource=='chat':resp=c.post('/api/chat',json={'workspace_id':w,'session_id':s,'request_id':'cross','content':'泄漏'})
    elif resource=='repo-register':resp=c.post('/api/repositories',json={'workspace_id':w,'root':'/etc','request_id':'cross'})
    elif resource=='job-retry':resp=c.post('/api'+paths[resource],params=params,json={})
    else:resp=c.get('/api'+paths[resource],params=params)
    assert resp.status_code in (403,404),resp.text
    assert '清茶' not in resp.text


def test_multi_route_files_graph_and_conversation_filter(env):
    p,c=env;a=register(c);o=a['user']['id'];w=a['workspaces'][0]['id'];s=session(c,w);s2=session(c,w)
    a=c.post('/api/attachments',data={'workspace_id':w,'session_id':s},files={'file':('sample.py',b'def answer(value):\n    return value + 1\n','text/plain')}).json()
    r=add(c,w,s,attachment_ids=[a['id']],text='昨天修复了数据库幂等问题，现在计划补充并发回归测试。')
    assert set(r['routes'])=={'conversation','engineering','source_code'};ingest(p)
    sources=p.session_sources(p.store.scope(o,w,'textual'),s)[0]
    assert next(m for m in sources if m.role=='attachment').metadata['routes']==['source_code']
    jobs=p.store.rows("SELECT * FROM cs_jobs WHERE id LIKE ? AND kind='derive'",(r['ingestion_id']+'%',))
    parse=next(j for j in jobs if json.loads(j['payload']).get('route')=='source_code');parse['payload']=json.loads(parse['payload']);p.do_derive(parse)
    graph=c.get('/api/memory-graph',params={'workspace_id':w,'session_id':s}).json()
    assert len(graph['nodes'])==3 and any(e['category']=='provenance' for e in graph['edges'])
    assert all(e['source'] in {n['id'] for n in graph['nodes']} and e['target'] in {n['id'] for n in graph['nodes']} for e in graph['edges'])
    add(c,w,s2,'other','另一个对话里的特别机密 KiwiQ38');ingest(p)
    result=p.search(o,w,'KiwiQ38','textual',retrieval_scope='workspace',session_id=s,conversation_first=True)
    assert all(m['session_id']!=s2 for m in result['items'])
    assert c.get('/api/memory-graph',params={'workspace_id':w,'session_id':s2,'center':r['memory_ids'][0]}).status_code==404
    assert c.get('/api/sessions/'+s+'/timeline',params={'workspace_id':w}).json()['total']==1


@pytest.mark.parametrize('filename,content,status',[('.env',b'password=abc',415),('file.pdf',b'%PDF-text',415),('bad.json',b'{bad',422),('binary.py',b'\x00code',415),('key.txt',b'api_key=secret1234567890',422),('latin.md',b'\xff\xfe',415)])
def test_attachment_validation(env,filename,content,status):
    p,c=env;w=register(c)['workspaces'][0]['id'];s=session(c,w)
    assert c.post('/api/attachments',data={'workspace_id':w,'session_id':s},files={'file':(filename,content,'text/plain')}).status_code==status


def test_deleted_conversation_invalidates_graph_search_sources_and_media(env):
    p,c=env;a=register(c);o=a['user']['id'];w=a['workspaces'][0]['id'];s=session(c,w);r=add(c,w,s);ingest(p)
    image=(Path(__file__).parents[2]/'demo/status_board.png').read_bytes();media=c.post('/api/media',data={'workspace_id':w,'session_id':s},files={'file':('board.png',image,'image/png')}).json()['id']
    scope=p.store.scope(o,w,'textual');derived=CanonicalMemory('derived-v3','茶偏好',session_id=s,role='derived',metadata={'source_versions':{r['memory_ids'][0]:1}},memory_type='profile',memory_level='L2_Knowledge')
    p.coordinator.apply(scope,'derive','v3',Mutation(memories=(derived,)),{})
    c.delete('/api/sessions/'+s,params={'workspace_id':w,'confirm':True})
    assert not p.eligible(scope,derived)
    assert c.get('/api/memories/'+r['memory_ids'][0],params={'workspace_id':w}).status_code==404
    assert c.get('/api/media/'+media,params={'workspace_id':w}).status_code==404
    assert not c.get('/api/memory-graph',params={'workspace_id':w,'retrieval_scope':'conversations'}).json()['nodes']


def test_explicit_legacy_claim_is_reversible_without_uid_changes(env):
    p,c=env;legacy='e'*32;w=p.store.create_workspace(legacy,'old guest')['id'];s=p.store.create_session(legacy,w,'textual','old')['id'];r=p.store.accept(legacy,AddMemory(workspace_id=w,session_id=s,request_id='legacy',messages=[{'content':'legacy remains'}]));ingest(p)
    a=register(c);assert c.get('/api/sessions',params={'workspace_id':w}).status_code==404
    with p.state.transaction() as db:db.execute('INSERT INTO cs_account_owners VALUES (?,?,?)',(a['user']['id'],legacy,1))
    assert c.get('/api/memories/'+r['memory_ids'][0],params={'workspace_id':w}).status_code==200
    with p.state.transaction() as db:db.execute('DELETE FROM cs_account_owners WHERE owner=?',(legacy,))
    assert c.get('/api/memories/'+r['memory_ids'][0],params={'workspace_id':w}).status_code==404
    assert p.locate(legacy,w,r['memory_ids'][0])[1].text_content=='legacy remains'


def test_cookie_flags_json_csrf_login_and_disabled_account(env):
    p,c=env;p.settings.secure_cookie=True
    c.post('/api/auth/register',json={'username':'secure-user','password':'correct-password-123'})
    r=c.post('/api/auth/login',json={'username':'secure-user','password':'correct-password-123'})
    header=r.headers['set-cookie'];assert 'HttpOnly' in header and 'Secure' in header and 'SameSite=strict' in header
    assert c.post('/api/auth/login',data={'username':'secure-user','password':'correct-password-123'}).status_code==403
    assert c.post('/api/auth/login',json={'username':'secure-user','password':'correct-password-123'},headers={'X-Mandol-Client':''}).status_code==403
    with p.state.transaction() as db:db.execute("UPDATE cs_accounts SET status='disabled'")
    assert p.accounts.authenticate(c.cookies[COOKIE]) is None


def test_v3_recover_multi_route_job_and_unknown_remote_without_replay(tmp_path):
    from mandol_competition.chinasoft.store import now
    p=Platform(Settings(data_dir=tmp_path));o='a'*32;w=p.store.create_workspace(o,'recover')['id'];s=p.store.create_session(o,w,'textual','新会话',unified=True)['id']
    receipt=p.store.accept(o,AddMemory(workspace_id=w,session_id=s,request_id='v3-crash',messages=[{'content':'我已修复数据库幂等问题，并完成一次回归测试。'}]))
    ingest(p);job=p.store.claim('derive','crashed');assert job
    with p.state.transaction() as db:db.execute('INSERT INTO cs_calls VALUES (?,?,?,?,?,?,?)',('unknown-call',job['id'],'running',None,None,now(),now()))
    p.close();p=Platform(Settings(data_dir=tmp_path));p.store.recover()
    assert p.store.rows('SELECT state FROM cs_calls WHERE id=?',('unknown-call',))[0]['state']=='unknown'
    assert p.store.ingestion(o,receipt['ingestion_id'])['status']=='saved'
    assert p.store.accept(o,AddMemory(workspace_id=w,session_id=s,request_id='v3-crash',messages=[{'content':'我已修复数据库幂等问题，并完成一次回归测试。'}]))==receipt
    assert p.store.rows("SELECT COUNT(*) n FROM cs_jobs WHERE ingestion=?",(receipt['ingestion_id'],))[0]['n']==4
    p.close()


def test_v3_concurrent_durable_retries_do_not_duplicate(env):
    from concurrent.futures import ThreadPoolExecutor
    p,c=env;a=register(c);o=a['user']['id'];w=a['workspaces'][0]['id'];s=session(c,w)
    payload=AddMemory(workspace_id=w,session_id=s,request_id='parallel',messages=[{'content':'昨天修复数据库事务问题。'}])
    with ThreadPoolExecutor(max_workers=4) as pool:receipts=list(pool.map(lambda _:p.store.accept(o,payload),range(12)))
    assert all(r==receipts[0] for r in receipts);ingest(p)
    assert p.store.rows('SELECT COUNT(*) n FROM cs_ingestions WHERE owner=?',(o,))[0]['n']==1
    assert len(MemoryViews(p).graph(o,w,s)['nodes'])==1


def test_rejected_profile_does_not_discard_other_validated_outputs(env,monkeypatch):
    from mandol_competition.chinasoft.highlevel import ConversationMemoryV3
    from mandol_competition.textual.core.hlm_production import CanonicalHighLevelMemory
    p,c=env
    prepared={'outputs':{'entity':[{'id':'already-audited'}],'profile':[]},'errors':{'profile':'Missing applicability qualifier'}}
    monkeypatch.setattr(CanonicalHighLevelMemory,'prepare',lambda *args:prepared)
    assert p.v3_hlm.prepare(None,None,None,None) is prepared
    assert not prepared['outputs']['profile']


def test_linked_repository_raw_revision_keeps_original_scope(env):
    p,c=env;a=register(c);o=a['user']['id'];w=a['workspaces'][0]['id'];s=session(c,w)
    with p.state.transaction() as db:
        db.execute("INSERT INTO cs_repositories(id,owner,workspace,name,root,status,created_at) VALUES ('repo-test',?,?,? ,?,'ready',1)",(o,w,'repo','/nonexistent'))
    c.patch('/api/sessions/'+s,params={'workspace_id':w},json={'repository_id':'repo-test'})
    r=add(c,w,s);ingest(p);scope,raw,_=p.locate(o,w,r['memory_ids'][0]);assert scope.user_id.endswith('.textual')
    revised=c.put('/api/memories/'+raw.logical_uid,params={'workspace_id':w},json={'request_id':'linked-revision','expected_version':1,'content':'修订后的真实用户偏好。'});assert revised.status_code==202,revised.text
    ingest(p);assert p.state.get_memories(scope,[raw.logical_uid])[raw.logical_uid].version==2


def test_engineering_sources_do_not_mix_when_conversation_changes_repository(env,monkeypatch):
    from mandol_competition.code.experience import ExperienceBuilder
    p,c=env;a=register(c);o=a['user']['id'];w=a['workspaces'][0]['id'];s=session(c,w)
    for i in ('a','b'):
        with p.state.transaction() as db:db.execute("INSERT INTO cs_repositories(id,owner,workspace,name,root,status,created_at) VALUES (?,?,?,?,?,'ready',1)",('repo-'+i,o,w,i,'/nonexistent/'+i))
        c.patch('/api/sessions/'+s,params={'workspace_id':w},json={'repository_id':'repo-'+i})
        r=add(c,w,s,'engineering-'+i,'我已修复 '+i+' 仓库的数据库事务问题。');ingest(p)
    job=p.store.rows('SELECT * FROM cs_jobs WHERE id=?',(r['ingestion_id']+':derive:engineering',))[0];job['payload']=json.loads(job['payload'])
    seen=[]
    def capture(self,scope,session,request_id,payload_digest,sources):seen.extend(sources);return Mutation()
    monkeypatch.setattr(ExperienceBuilder,'prepare',capture);p.do_derive(job)
    assert seen and all(m.metadata['repository_id']=='repo-b' for m in seen)


def test_old_raw_request_identity_survives_unified_upgrade(env):
    p,c=env;a=register(c);o=a['user']['id'];w=a['workspaces'][0]['id'];s=p.store.create_session(o,w,'code','old code session')['id'];data=AddMemory(workspace_id=w,session_id=s,request_id='old-replay',mode='code',messages=[{'content':'修复原始代码记录，保持幂等。'}]);receipt=p.store.accept(o,data);ingest(p)
    c.patch('/api/sessions/'+s,params={'workspace_id':w},json={'unified':True})
    assert p.store.accept(o,data)==receipt
    assert p.store.rows('SELECT COUNT(*) n FROM cs_ingestions WHERE session=?',(s,))[0]['n']==1
    with pytest.raises(Exception):p.store.accept(o,AddMemory(workspace_id=w,session_id=s,request_id='old-replay',messages=[{'content':'different'}]))


def test_registration_does_not_disclose_existing_account(env):
    p,c=env
    payload={'username':'private-account','password':'correct-password-123'}
    first=c.post('/api/auth/register',json=payload);again=c.post('/api/auth/register',json={**payload,'password':'other-password-123'})
    assert first.status_code==again.status_code==201 and first.json()==again.json()
    assert p.accounts.login('private-account','correct-password-123','test')[2]['role']=='user'
    with pytest.raises(Exception):p.accounts.login('private-account','other-password-123','test')


def test_engineering_family_is_semantic_even_with_legacy_default_role(env):
    p,c=env;a=register(c);o=a['user']['id'];w=a['workspaces'][0]['id'];s=session(c,w);r=add(c,w,s);ingest(p);scope=p.store.scope(o,w,'textual')
    derived=CanonicalMemory('legacy-role-experience','经验',metadata={'memory_family':'code','source_versions':{r['memory_ids'][0]:1}},session_id=s,memory_type='engineering_experience',memory_level='L1_Episode')
    p.coordinator.apply(scope,'semantic-family','family',Mutation(memories=(derived,)),{})
    data=p.serialize(scope,derived)
    assert data['memory_family']=='code' and data['mode']=='code' and data['source_kind']=='derived'


def test_chunked_requests_cannot_bypass_body_limits(env):
    p,c=env
    # No Content-Length: the ASGI receive path must enforce the actual bytes.
    r=c.post('/api/auth/login',content=iter([b'{"username":"'+b'x'*20000,b'","password":"never"}']),headers={'Content-Type':'application/json'})
    assert r.status_code==413,r.text
    a=register(c);w=a['workspaces'][0]['id'];s=session(c,w)
    content=json.dumps({'workspace_id':w,'session_id':s,'request_id':'oversized','messages':[{'content':'x'*400000}]}).encode()
    r=c.post('/api/memories',content=iter([content]),headers={'Content-Type':'application/json'})
    assert r.status_code==413,r.text
    assert not p.store.rows('SELECT 1 FROM cs_ingestions')
