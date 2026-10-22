"""Real SQLite/API/Tree-sitter tests; remote fault/commit fixtures are explicit.

Live provider, GPU, LLM and browser evidence is produced by v32 acceptance
scripts separately. These fixtures do not claim external network success.
"""
from pathlib import Path
import hashlib,json,socket,time
import pytest
from fastapi.testclient import TestClient
from mandol_competition.chinasoft.api import create_app
from mandol_competition.chinasoft.config import Settings
from mandol_competition.chinasoft.service import Platform
from mandol_competition.chinasoft.auth import COOKIE,hashed
from mandol_competition.chinasoft.remote_git import validate_url,public_addresses,PublicHTTP,PinnedHTTPS
from mandol_competition.chinasoft.store import AppError,now

PASSWORD='v32-secure-password-123'

@pytest.fixture
def env(tmp_path):
    p=Platform(Settings(data_dir=tmp_path,sparse_url='',allow_retrieval_degraded=True,registration_enabled=True,registration_approval=False))
    with TestClient(create_app(platform=p,start_workers=False)) as c:
        c.headers['X-Mandol-Client']='studio'
        yield p,c


def login(c,name):
    c.cookies.clear();r=c.post('/api/auth/login',json={'username':name,'password':PASSWORD});assert r.status_code==200,r.text
    value=r.json();c.headers['X-CSRF-Token']=value['csrf_token'];return value


def user(p,c,name='person-alpha',admin=False):
    p.accounts.create(name,PASSWORD,admin=admin,approved=True)
    return login(c,name)


def test_registration_atomic_private_confirmation_injection_and_login_rotation(env):
    p,c=env
    for extra in ({'role':'admin'},{'is_admin':True},{'password_confirmation':'different'}):
        assert c.post('/api/auth/register',json={'username':'new-person','password':PASSWORD,**extra}).status_code==422
    assert not p.store.rows('SELECT * FROM cs_accounts')
    assert c.post('/api/auth/register',json={'username':'new-person','password':'aaaaaaaaaaaa'}).status_code==422
    assert c.post('/api/auth/register',json={'username':'new-person','password':PASSWORD,'password_confirmation':PASSWORD}).status_code==201
    identity=login(c,'new-person');first=c.cookies[COOKIE]
    assert identity['user']['role']=='user' and len(identity['workspaces'])==1 and identity['workspaces'][0]['name']=='我的记忆工作区'
    assert p.store.rows('SELECT * FROM cs_accounts')[0]['password_hash'].startswith('$argon2id$')
    again=c.post('/api/auth/login',json={'username':'new-person','password':PASSWORD});assert again.status_code==200
    assert c.cookies[COOKIE]!=first and not p.accounts.authenticate(first)
    assert p.store.rows('SELECT last_login_at FROM cs_accounts')[0]['last_login_at']


def test_registration_global_capacity_and_rate_limit(env):
    p,c=env;p.settings.max_accounts=1
    assert c.post('/api/auth/register',json={'username':'one-person','password':PASSWORD}).status_code==201
    assert c.post('/api/auth/register',json={'username':'two-person','password':PASSWORD}).status_code==429
    p.settings.registration_enabled=False
    assert c.post('/api/auth/register',json={'username':'three-person','password':PASSWORD}).status_code==403
    p.settings.registration_enabled=True
    for _ in range(6):r=c.post('/api/auth/register',json={'username':'four-person','password':PASSWORD})
    assert r.status_code==429


@pytest.mark.parametrize('path', ['/admin/users','/admin/overview','/admin/audit','/admin/users/guess/resources'])
def test_admin_get_is_not_authorized_by_frontend_visibility(env,path):
    p,c=env;user(p,c)
    assert c.get('/api'+path).status_code==403


@pytest.mark.parametrize('path,method,data',[
 ('/admin/users','POST',{'username':'fake-admin','password':PASSWORD}),
 ('/admin/users/guess','PATCH',{'role':'admin','confirm':True}),
 ('/admin/users/guess/quotas','PUT',{'limits':{}}),
 ('/admin/default-quotas','PUT',{'limits':{}})])
def test_ordinary_users_cannot_mutate_admin_control_plane(env,path,method,data):
    p,c=env;user(p,c)
    assert c.request(method,'/api'+path,json=data).status_code==403


def test_admin_actions_last_admin_sessions_audit_and_private_content(env):
    p,c=env;admin=user(p,c,'platform-admin',True);aid=admin['user']['id']
    created=c.post('/api/admin/users',json={'username':'managed-person','password':PASSWORD});assert created.status_code==201
    target=created.json()['id'];w=p.accounts.workspaces(target)[0]['id'];s=p.store.create_session(target,w,'textual','PRIVATE-NEVER-ADMIN-CONTENT',unified=True)
    assert c.get('/api/sessions',params={'workspace_id':w}).status_code==404
    listing=c.get('/api/admin/users',params={'q':'managed','role':'user','status':'active','limit':1}).json()
    assert listing['total']==1 and listing['items'][0]['usage']['conversations']==1
    assert not any(word in json.dumps(listing) for word in ('password_hash','token_hash','PRIVATE-NEVER'))
    assert c.patch('/api/admin/users/'+aid,json={'status':'disabled','confirm':True}).status_code==409
    assert c.patch('/api/admin/users/'+target,json={'status':'disabled'}).status_code==409
    old=login(c,'managed-person');token=c.cookies[COOKIE]
    login(c,'platform-admin')
    assert c.patch('/api/admin/users/'+target,json={'status':'disabled','confirm':True}).status_code==200
    assert not p.accounts.authenticate(token)
    assert c.patch('/api/admin/users/'+target,json={'status':'active','confirm':True}).status_code==200
    token=login(c,'managed-person');cookie=c.cookies[COOKIE];login(c,'platform-admin')
    assert c.patch('/api/admin/users/'+target,json={'revoke_sessions':True,'confirm':True}).status_code==200
    assert not p.accounts.authenticate(cookie)
    assert c.patch('/api/admin/users/'+target,json={'role':'admin','confirm':True}).status_code==403
    assert c.patch('/api/admin/users/'+target,json={'role':'admin','confirm':True,'current_password':PASSWORD}).status_code==200
    assert p.store.rows('SELECT role FROM cs_accounts WHERE id=?',(target,))[0]['role']=='admin'
    assert c.patch('/api/admin/users/'+target,json={'role':'user','confirm':True,'current_password':PASSWORD}).status_code==200
    events=c.get('/api/admin/audit').json()['items']
    assert {'account_create','account_status','account_role','sessions_revoke'}<={e['action'] for e in events}
    assert any(e['result']=='denied' for e in events)
    assert PASSWORD not in json.dumps(events)


def test_user_and_platform_quotas_enforced_and_reset(env):
    p,c=env;admin=user(p,c,'platform-admin',True)
    normal=user(p,c);target=normal['user']['id'];w=normal['workspaces'][0]['id']
    login(c,'platform-admin')
    assert c.put('/api/admin/users/'+target+'/quotas',json={'limits':{'workspaces':1,'conversations':1,'storage_bytes':1024,'model_per_day':1}}).status_code==200
    assert c.put('/api/admin/users/'+target+'/quotas',json={'limits':{'unbounded':-1}}).status_code==422
    login(c,'person-alpha')
    assert c.post('/api/workspaces',json={'name':'over-quota'}).status_code==429
    assert c.post('/api/sessions',json={'workspace_id':w}).status_code==201
    assert c.post('/api/sessions',json={'workspace_id':w}).status_code==429
    s=c.get('/api/sessions',params={'workspace_id':w}).json()['items'][0]['id']
    assert c.post('/api/attachments',data={'workspace_id':w,'session_id':s},files={'file':('a.md',b'a'*1100,'text/plain')}).status_code==429
    marker=p.generation.account_context.set(target)
    try:
        p.generation.charge()
        with pytest.raises(Exception) as exc:p.generation.charge()
        assert exc.value.code=='quota_exceeded'
    finally:p.generation.account_context.reset(marker)
    login(c,'platform-admin')
    assert c.put('/api/admin/users/'+target+'/quotas',json={'limits':None}).status_code==200
    assert not c.get('/api/admin/users/'+target+'/resources').json()['overrides']
    assert c.put('/api/admin/default-quotas',json={'limits':{'workspaces':1}}).status_code==200
    login(c,'person-alpha');assert c.post('/api/workspaces',json={'name':'platform quota'}).status_code==429


@pytest.mark.parametrize('url',[
 'http://github.com/o/r','file:///etc/passwd','git@github.com:o/r','ssh://github.com/o/r',
 'https://127.0.0.1/o/r','https://169.254.169.254/latest/meta-data','https://192.168.1.1/o/r',
 'https://github.com.evil.invalid/o/r','https://user:pass@github.com/o/r','https://github.com:443/o/r',
 'https://github.com/o/r?x=--upload-pack','https://github.com/o/../etc','https://github.com/o/%2e%2e',
 'https://github.com/o/r#main','https://github.com/o/r/tree/main','https://gitcode.com/o/r',
 'https://github.com/o/r\\x','https://github.com/o/r\n','https://github.com/o//r'])
def test_remote_url_rejects_unsafe_or_unimplemented_targets(url):
    with pytest.raises((AppError,ValueError)):validate_url(url)


@pytest.mark.parametrize('ref',['--upload-pack=evil','../main','main;rm -rf /','x@{y}','a//b','refs/x.lock'])
def test_remote_ref_is_data_and_rejects_unsafe_values(ref):
    with pytest.raises(AppError):validate_url('https://github.com/o/r',ref)


@pytest.mark.parametrize('address',['127.0.0.1','10.0.0.1','169.254.169.254','::1','::ffff:127.0.0.1','192.0.2.1','224.0.0.1','ff00::1'])
def test_dns_private_and_mixed_resolutions_rejected(monkeypatch,address):
    monkeypatch.setattr(socket,'getaddrinfo',lambda *a,**k:[(socket.AF_INET,socket.SOCK_STREAM,6,'',('8.8.8.8',443)),(socket.AF_INET,socket.SOCK_STREAM,6,'',(address,443))])
    with pytest.raises(AppError) as exc:public_addresses('api.github.com')
    assert exc.value.code=='git_ssrf_rejected'


def test_redirect_is_rejected_without_second_connection(monkeypatch):
    from mandol_competition.chinasoft import remote_git
    calls=[]
    class Connection:
        def __init__(self,*args):calls.append(args)
        def request(self,*a,**k):pass
        def getresponse(self):return type('Response',(),{'status':302})()
        def close(self):pass
    monkeypatch.setattr(remote_git,'public_addresses',lambda host:['8.8.8.8'])
    monkeypatch.setattr(remote_git,'PinnedHTTPS',Connection)
    with pytest.raises(AppError) as exc:PublicHTTP(time.monotonic()+10,lambda:None).get('https://api.github.com/repos/a/b')
    assert exc.value.code=='git_redirect_rejected' and len(calls)==1


def test_tls_connection_pins_validated_address_and_keeps_original_sni(monkeypatch):
    calls=[];connection=PinnedHTTPS('api.github.com','8.8.8.8',2)
    class Context:
        def wrap_socket(self,sock,server_hostname):calls.append(('sni',server_hostname));return sock
    connection._context=Context()
    monkeypatch.setattr(socket,'create_connection',lambda target,timeout:(calls.append(('connect',target)),object())[1])
    connection.connect()
    assert calls==[('connect',('8.8.8.8',443)),('sni','api.github.com')]


def fixture_provider(monkeypatch,p,content):
    """Explicit remote provider fixture; production scanners/storage stay real."""
    from mandol_competition.chinasoft.remote_git import RemoteGit
    def metadata(self,http,repo):
        entries=[]
        for path,data in content.items():
            entries.append({'path':path,'type':'blob','mode':'100644','size':len(data),'sha':hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()})
        entries.extend([{'path':'linked.py','type':'blob','mode':'120000','sha':'a'*40},{'path':'module','type':'commit','mode':'160000','sha':'a'*40}])
        return hashlib.sha1(repr(content).encode()).hexdigest(),'main',entries,lambda e:'https://raw.githubusercontent.com/fixture/'+e['path']
    monkeypatch.setattr(RemoteGit,'metadata',metadata)
    monkeypatch.setattr(PublicHTTP,'get',lambda self,url,**k:content[url.split('/fixture/',1)[1]])


def scan(p):
    job=p.store.claim('repository','v32-test');assert job
    result=p.do_repository(job);p.store.finish(job,result);return result,job


def test_remote_real_storage_parser_increment_delete_restart_and_source_isolation(env,monkeypatch):
    p,c=env;first=user(p,c);owner=first['user']['id'];w=first['workspaces'][0]['id']
    data={'src/a.py':b'def alpha():\n    return 1\n','src/use.py':b'from src.a import alpha\n',
          '.env':b'not-read','vendor/big.py':b'bad','binary.py':b'\0binary',
          'pointer.py':b'version https://git-lfs.github.com/spec/v1\noid sha256:aaa\n',
          'huge.py':b'a'*(256*1024+1)}
    fixture_provider(monkeypatch,p,data)
    body={'workspace_id':w,'url':'https://github.com/example/fixture','request_id':'initial'}
    r=c.post('/api/repositories/remote',json=body);assert r.status_code==202,r.text;repo=r.json()['id']
    assert c.post('/api/repositories/remote',json=body).json()['job_id']==r.json()['job_id']
    assert c.post('/api/repositories/remote',json={**body,'request_id':'parallel'}).status_code==429
    result,job=scan(p);assert result['files']==2 and result['symbols']==1 and result['skipped_count']>=6
    assert len(p.repositories.symbols(owner,w,repo,0,20)['items'])==1
    first_file=p.store.rows('SELECT * FROM cs_repository_files WHERE repository=? AND path=?',(repo,'src/use.py'))[0]
    assert c.post('/api/repositories/'+repo+'/scan',params={'workspace_id':w},json={'request_id':'noop'}).status_code==202
    result,_=scan(p);assert result['changed_files']==0 and result['unchanged_files']==2
    assert p.store.repository(owner,w,repo)['generation']==1
    assert p.explorer(owner,w,'code',repository_id=repo,memory_type='repository_map')['items']
    data['src/a.py']=b'def beta():\n    return 2\n'
    assert c.post('/api/repositories/'+repo+'/scan',params={'workspace_id':w},json={'request_id':'change'}).status_code==202
    result,_=scan(p);assert result['changed_files']==1
    assert p.repositories.symbols(owner,w,repo,0,20)['items'][0]['name']=='beta'
    after=p.store.rows('SELECT * FROM cs_repository_files WHERE repository=? AND path=?',(repo,'src/use.py'))[0]
    assert after['version']==first_file['version'] and after['uids']==first_file['uids']
    del data['src/a.py']
    c.post('/api/repositories/'+repo+'/scan',params={'workspace_id':w},json={'request_id':'delete'});result,_=scan(p)
    assert result['deleted_files']==1 and not p.repositories.symbols(owner,w,repo,0,20)['items']
    other=user(p,c,'person-beta');w2=other['workspaces'][0]['id']
    assert c.get('/api/repositories/'+repo,params={'workspace_id':w}).status_code==404
    second=c.post('/api/repositories/remote',json={**body,'workspace_id':w2,'request_id':'other'}).json()
    assert second['id']!=repo
    # Restart recovery is exercised on a real durable claimed job.
    claimed=p.store.claim('repository','crash-fixture');assert claimed
    p.store.recover();recovered=p.store.claim('repository','resume-fixture');assert recovered['id']==claimed['id'] and recovered['attempts']==2
    p.store.finish(recovered,p.do_repository(recovered))
    login(c,'person-alpha')
    assert c.delete('/api/repositories/'+repo,params={'workspace_id':w,'confirm':True}).status_code==200
    assert c.get('/api/repositories/'+repo,params={'workspace_id':w}).status_code==404
    assert not p.store.rows("SELECT * FROM cs_sessions WHERE repository=? AND deleted=0",(repo,))


def test_remote_limits_cancel_and_ordinary_local_bypass(env,monkeypatch):
    p,c=env;identity=user(p,c);w=identity['workspaces'][0]['id']
    with p.state.transaction() as db:db.execute('UPDATE cs_accounts SET local_repository=1')
    assert c.post('/api/repositories',json={'workspace_id':w,'root':str(Path.cwd()),'request_id':'local'}).status_code==403
    data={'a.py':b'def a():\n return 1\n','b.py':b'def b():\n return 2\n'}
    fixture_provider(monkeypatch,p,data);p.settings.remote_git_max_files=1
    r=c.post('/api/repositories/remote',json={'workspace_id':w,'url':'https://github.com/example/budget','request_id':'limited'}).json()
    job=p.store.claim('repository','budget')
    with pytest.raises(AppError) as exc:p.do_repository(job)
    assert exc.value.code=='git_repository_limit';p.store.fail(job,exc.value)
    assert not list((p.settings.data_dir/'remote-repositories').glob('**/tmp-*'))
    p.settings.remote_git_max_files=500
    r=c.post('/api/repositories/remote',json={'workspace_id':w,'url':'https://github.com/example/cancel','request_id':'cancel'}).json()
    assert c.post('/api/jobs/'+r['job_id']+'/cancel',params={'workspace_id':w},json={}).status_code==202
    assert not p.store.claim('repository','none')


def test_migration_is_additive_and_idempotent_for_populated_v31(env):
    p,c=env;identity=user(p,c);owner=identity['user']['id'];w=identity['workspaces'][0]['id']
    s=p.store.create_session(owner,w,'textual','existing history',unified=True)
    from mandol_competition.chinasoft.migrations import migrate
    original=p.store.rows('SELECT id,owner,workspace,title FROM cs_sessions')
    # Simulate the pre-V3.2 schema on a populated isolated SQLite database.
    with p.state.transaction() as db:
        db.execute('DELETE FROM cs_schema_migrations WHERE version=32')
        db.execute('DROP INDEX cs_repository_owner')
        db.execute('ALTER TABLE cs_accounts DROP COLUMN last_login_at')
        for name in ('source_kind','source_url','source_ref','deleted'):db.execute('ALTER TABLE cs_repositories DROP COLUMN '+name)
        for name in ('result','details'):db.execute('ALTER TABLE cs_admin_audit DROP COLUMN '+name)
        for table in ('cs_account_quotas','cs_quota_defaults','cs_usage_buckets','cs_remote_snapshots'):db.execute('DROP TABLE '+table)
    migrate(p.state);migrate(p.state)
    assert p.store.rows('SELECT id,owner,workspace,title FROM cs_sessions')==original
    assert p.accounts.workspaces(owner)[0]['id']==w
    assert p.store.rows('PRAGMA integrity_check')[0]['integrity_check']=='ok'


def test_linked_remote_conversation_indexes_its_own_raw_scope_and_legacy_job(env,monkeypatch):
    p,c=env;identity=user(p,c);owner=identity['user']['id'];w=identity['workspaces'][0]['id']
    fixture_provider(monkeypatch,p,{'a.py':b'def a():\n return 1\n'})
    repo=c.post('/api/repositories/remote',json={'workspace_id':w,'url':'https://github.com/example/index','request_id':'index'}).json()['id'];scan(p)
    s=p.store.create_session(owner,w,'textual','linked',repo,True)
    result=c.post('/api/memories',json={'workspace_id':w,'session_id':s['id'],'request_id':'linked-raw','messages':[{'content':'仓库是附加知识源，原始对话仍属于当前对话。'}]}).json()
    job=p.store.claim('ingest','v32-index');p.store.finish(job,p.do_ingest(job))
    row=p.store.rows('SELECT * FROM cs_jobs WHERE id=?',(result['jobs'][1],))[0];payload=json.loads(row['payload'])
    assert payload['repository_id']=='' and payload['associated_repository_id']==repo
    monkeypatch.setattr(p.coordinator,'project_optional',lambda scope:True)
    # Old durable payloads must also recover without rewriting canonical truth.
    old={**row,'payload':{**payload,'repository_id':repo}}
    assert p.do_index(old)['scopes']==1
    assert p.store.repository(owner,w,repo)['status']=='lexical_ready'


def test_foreign_ids_cannot_be_read_using_an_owned_workspace(env,monkeypatch):
    p,c=env;a=user(p,c);aw=a['workspaces'][0]['id'];as_=p.store.create_session(a['user']['id'],aw,'textual','A',unified=True)['id']
    b=user(p,c,'person-beta');bw=b['workspaces'][0]['id'];bs=p.store.create_session(b['user']['id'],bw,'textual','B',unified=True)['id']
    r=c.post('/api/memories',json={'workspace_id':bw,'session_id':bs,'request_id':'private','messages':[{'content':'B-PRIVATE-CONTENT-66521'}]}).json()
    job=p.store.claim('ingest','foreign-ids');p.store.finish(job,p.do_ingest(job));memory=r['memory_ids'][0]
    image=c.post('/api/media',data={'workspace_id':bw,'session_id':bs},files={'file':('x.png',(Path(__file__).parents[2]/'demo/status_board.png').read_bytes(),'image/png')}).json()['id']
    attachment=c.post('/api/attachments',data={'workspace_id':bw,'session_id':bs},files={'file':('b.py',b'def private_beta():\n return 3\n','text/plain')}).json()['id']
    fixture_provider(monkeypatch,p,{'a.py':b'def secret_beta():\n return 1\n'})
    repo=c.post('/api/repositories/remote',json={'workspace_id':bw,'url':'https://github.com/example/foreign','request_id':'foreign'}).json()['id'];scan(p)
    login(c,'person-alpha')
    paths=['/sessions/'+bs,'/memories/'+memory,'/memories/'+memory+'/relations','/media/'+image,'/attachments/'+attachment,
           '/repositories/'+repo,'/repositories/'+repo+'/files','/repositories/'+repo+'/symbols','/repositories/'+repo+'/map','/repositories/'+repo+'/history',
           '/sessions/'+bs+'/timeline','/ingestions/'+r['ingestion_id']]
    for path in paths:
        response=c.get('/api'+path,params={'workspace_id':aw,'path':'a.py'})
        assert response.status_code==404,(path,response.text)
        assert 'B-PRIVATE' not in response.text
    assert c.get('/api/memory-graph',params={'workspace_id':aw,'session_id':as_,'center':memory}).status_code==404
    assert c.post('/api/chat',json={'workspace_id':aw,'session_id':bs,'request_id':'foreign','content':'guess'}).status_code==404
    assert c.post('/api/memories',json={'workspace_id':aw,'session_id':as_,'request_id':'foreign','attachment_ids':[attachment],'messages':[{'content':'use foreign file'}]}).status_code==404
    assert c.post('/api/jobs/'+r['jobs'][0]+'/retry',params={'workspace_id':aw},json={}).status_code==404


def test_remote_response_byte_limit_and_timeout_are_real_client_errors(monkeypatch):
    from mandol_competition.chinasoft import remote_git
    class Response:
        status=200
        def getheader(self,name,default=None):return '10485760' if name=='Content-Length' else default
        def read(self,*a):raise AssertionError('oversize body must not be read')
    class Connection:
        def __init__(self,*a):pass
        def request(self,*a,**k):pass
        def getresponse(self):return Response()
        def close(self):pass
    monkeypatch.setattr(remote_git,'public_addresses',lambda host:['8.8.8.8'])
    monkeypatch.setattr(remote_git,'PinnedHTTPS',Connection)
    with pytest.raises(AppError) as exc:PublicHTTP(time.monotonic()+10,lambda:None).get('https://api.github.com/repos/a/b',maximum=1024)
    assert exc.value.code=='git_response_limit'
    def timed_out(self,*a,**k):raise TimeoutError('explicit timeout fault injection')
    monkeypatch.setattr(Connection,'request',timed_out)
    with pytest.raises(AppError) as exc:PublicHTTP(time.monotonic()+10,lambda:None).get('https://api.github.com/repos/a/b')
    assert exc.value.code=='git_connection'


def test_snapshot_and_partial_parse_resume_without_duplicate_publication(env,monkeypatch):
    p,c=env;identity=user(p,c);w=identity['workspaces'][0]['id'];owner=identity['user']['id']
    fixture_provider(monkeypatch,p,{'a.py':b'def a():\n return 1\n','b.py':b'def b():\n return 2\n'})
    r=c.post('/api/repositories/remote',json={'workspace_id':w,'url':'https://github.com/example/resume','request_id':'resume'}).json()
    original=p.coordinator.apply;calls=0
    def crash(scope,key,*args,**kwargs):
        nonlocal calls
        calls+=1
        if calls==2:raise RuntimeError('explicit worker crash after first real SQLite publication')
        return original(scope,key,*args,**kwargs)
    monkeypatch.setattr(p.coordinator,'apply',crash)
    job=p.store.claim('repository','crashed')
    with pytest.raises(RuntimeError):p.do_repository(job)
    assert len(p.store.rows('SELECT * FROM cs_repository_files WHERE repository=?',(r['id'],)))==1
    monkeypatch.setattr(p.coordinator,'apply',original);p.store.recover()
    resumed=p.store.claim('repository','resumed');assert resumed['id']==job['id']
    p.store.finish(resumed,p.do_repository(resumed))
    assert p.repositories.files(owner,w,r['id'],0,20)['total']==2
    assert p.repositories.symbols(owner,w,r['id'],0,20)['total']==2
    assert not list((p.settings.data_dir/'remote-repositories').glob('**/tmp-*'))


def test_concurrent_registration_account_cap_has_no_orphan_workspace(env):
    from concurrent.futures import ThreadPoolExecutor
    p,c=env;p.settings.max_accounts=1
    def create(name):
        try:p.accounts.create(name,PASSWORD,approved=True);return True
        except AppError:return False
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(create,['cap-alpha','cap-beta']))
    assert sorted(results)==[False,True]
    for table in ('cs_accounts','cs_account_owners','cs_workspaces'):
        assert len(p.store.rows('SELECT * FROM '+table))==1


def test_upload_admission_precedes_multipart_and_retained_file_count_is_bounded(env):
    p,c=env;identity=user(p,c);account=identity['user']['id'];w=identity['workspaces'][0]['id'];s=p.store.create_session(account,w,'textual','uploads',unified=True)['id']
    p.store.access.set_quotas(p.accounts.create('quota-admin',PASSWORD,admin=True)['id'],account,{'uploads_per_minute':1,'attachment_files':1})
    first=c.post('/api/attachments',data={'workspace_id':w,'session_id':s},files={'file':('a.md',b'first source','text/plain')});assert first.status_code==201
    # Malformed multipart would be a parser error if admission happened too late.
    assert c.post('/api/attachments',content=b'not a multipart form',headers={'Content-Type':'multipart/form-data; boundary=bad'}).status_code==429
    with p.state.transaction() as db:db.execute("DELETE FROM cs_usage_buckets WHERE account=? AND action='uploads_per_minute'",(account,))
    assert c.delete('/api/attachments/'+first.json()['id'],params={'workspace_id':w}).status_code==200
    assert c.post('/api/attachments',data={'workspace_id':w,'session_id':s},files={'file':('b.md',b'second source','text/plain')}).status_code==429


def test_global_generation_budget_is_atomic_across_accounts(env):
    p,c=env;a=user(p,c);b=user(p,c,'person-beta');p.settings.model_requests_per_day=1
    token=p.generation.account_context.set(a['user']['id'])
    try:p.generation.charge()
    finally:p.generation.account_context.reset(token)
    token=p.generation.account_context.set(b['user']['id'])
    try:
        with pytest.raises(Exception) as exc:p.generation.charge()
        assert exc.value.code=='platform_model_quota'
    finally:p.generation.account_context.reset(token)
    assert p.store.access.summary(b['user']['id'])['usage']['model_per_day']==0


def test_github_blob_fallback_preserves_actual_blob_identity(env,monkeypatch):
    import base64
    p,c=env;identity=user(p,c);w=identity['workspaces'][0]['id'];content={'a.py':b'def fallback_symbol():\n return 1\n'}
    fixture_provider(monkeypatch,p,content);calls=[]
    def get(self,url,**kw):
        calls.append(url)
        if 'raw.githubusercontent.com' in url:raise AppError('explicit raw host network failure',503,code='git_connection')
        data=content['a.py'];sha=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        return {'sha':sha,'encoding':'base64','content':base64.b64encode(data).decode()}
    monkeypatch.setattr(PublicHTTP,'get',get)
    repo=c.post('/api/repositories/remote',json={'workspace_id':w,'url':'https://github.com/example/fallback','request_id':'fallback'}).json()['id']
    result,_=scan(p)
    assert result['github_blob_api_fallback'] and result['symbols']==1
    assert len(calls)==2 and calls[1].startswith('https://api.github.com/repos/example/fallback/git/blobs/')


def test_cancel_wins_over_simultaneous_remote_network_failure(env,monkeypatch):
    p,c=env;identity=user(p,c);w=identity['workspaces'][0]['id']
    r=c.post('/api/repositories/remote',json={'workspace_id':w,'url':'https://github.com/example/cancel-race','request_id':'race'}).json()
    def failing(job):
        with p.state.transaction() as db:db.execute("UPDATE cs_jobs SET error_code='cancel_requested' WHERE id=?",(job['id'],))
        p.stop_event.set()
        raise AppError('explicit concurrent network fault',503,code='git_connection')
    monkeypatch.setattr(p,'do_repository',failing);p.worker('repository')
    job=p.store.rows('SELECT * FROM cs_jobs WHERE id=?',(r['job_id'],))[0]
    assert job['status']=='cancelled' and job['error_code']=='repository_cancelled'
