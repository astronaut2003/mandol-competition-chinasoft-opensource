"""Real SQLite/HTTP deletion tests; no production workspace is cleared."""
from dataclasses import replace
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
from fastapi.testclient import TestClient
from mandol_substrate.state import Mutation, Edge
from mandol_competition.chinasoft.api import create_app
from mandol_competition.chinasoft.config import Settings
from mandol_competition.chinasoft.service import Platform
from mandol_competition.chinasoft.store import AppError
from mandol_competition.chinasoft.visualization import MemoryViews
from tests.chinasoft.auth_helpers import login_test
from tests.chinasoft.test_v33 import env
from tests.chinasoft.test_v4_retrieval import publish,raw,derived


def clear(client,workspace,request='clear-request',confirmation='清除所有记忆',**extra):
    return client.post('/api/workspaces/'+workspace+'/memories/clear',json={'confirmation':confirmation,'request_id':request,**extra})


@pytest.mark.parametrize('phrase',['','清除','清除所有記憶',' 清除所有记忆','清除所有记忆 ','清除所有记忆\n'])
def test_exact_confirmation_required_without_modification(env,phrase):
    p,c,o,w,s=env;scope=p.store.scope(o,w,'textual');publish(p,scope,raw('keep','Keep this source',s))
    assert clear(c,w,confirmation=phrase).status_code==400
    assert p.state.get_memories(scope,['keep'])['keep'].active
    assert not p.store.rows('SELECT * FROM cs_memory_clears')


def test_authorization_csrf_and_input_validation(env):
    p,c,o,w,s=env;scope=p.store.scope(o,w,'textual');publish(p,scope,raw('keep','A private record',s))
    assert clear(c,w,role='admin').status_code==422
    token=c.headers.pop('X-CSRF-Token');assert clear(c,w).status_code==403;c.headers['X-CSRF-Token']=token
    other=login_test(c,p)
    assert clear(c,w).status_code==404
    c.cookies.clear();assert clear(c,w).status_code==401
    assert p.state.get_memories(scope,['keep'])['keep'].active


def test_workspace_wide_basic_higher_and_edges_removed_other_scopes_retained(env):
    p,c,o,w,s=env;scope=p.store.scope(o,w,'textual')
    second=p.store.create_session(o,w,'textual','Another conversation',unified=True)['id']
    a=raw('first','A source',s);b=raw('second','B source',second);h=derived('higher','A summary',a,'profile')
    alias=replace(h,logical_uid='reference',memory_type='canonical_reference',memory_level='Reference',active=False)
    publish(p,scope,a,b,h,alias,edges=(Edge('support','higher','first','SUPPORTED_BY'),))
    visual=p.store.scope(o,w,'multimodal');publish(p,visual,raw('visual','A visual source',s))
    other_w=p.store.create_workspace(o,'Keep workspace')['id'];other_scope=p.store.scope(o,other_w,'textual');publish(p,other_scope,raw('keep','Independent workspace',s))
    response=clear(c,w);assert response.status_code==200,response.text
    assert response.json()['records']==4 and response.json()['basic']==3 and response.json()['higher']==1
    assert response.json()['canonical_records']==5
    assert all(not m.active and m.lifecycle=='deleted' for m in p.state.get_memories(scope,['first','second','higher','reference']).values())
    assert not p.state.get_memories(visual,['visual'])['visual'].active
    assert not p.state._rows('SELECT 1 FROM graph_edges WHERE scope=? AND active=1',(scope.key,))
    assert p.state.get_memories(other_scope,['keep'])['keep'].active
    assert c.get('/api/sessions',params={'workspace_id':w}).json()['total']>=2
    assert not MemoryViews(p).graph(o,w)['nodes']
    assert c.get('/api/memory-schema',params={'workspace_id':w,'retrieval_scope':'conversations'}).json()['instance']['valid']==0
    # Existing deletion audit is retained, not a claim of physical erasure.
    assert c.get('/api/memories/first',params={'workspace_id':w,'audit':True}).json()['quality']['state']=='deleted'


def test_replay_does_not_clear_new_memories_and_old_jobs_cannot_retry(env):
    p,c,o,w,s=env;scope=p.store.scope(o,w,'textual');publish(p,scope,raw('old','Old source',s))
    accepted=c.post('/api/memories',json={'workspace_id':w,'session_id':s,'request_id':'queued-input','messages':[{'content':'Queued memory'}]}).json()
    first=clear(c,w).json()
    assert first['cancelled_jobs']>=2
    assert p.store.claim('ingest','after-clear') is None
    for job in accepted['jobs']:
        assert p.store.rows('SELECT status FROM cs_jobs WHERE id=?',(job,))[0]['status']=='cancelled'
        with pytest.raises(AppError):p.store.retry(o,job)
    publish(p,scope,raw('new','New source after clearing',s))
    assert clear(c,w).json()==first
    assert p.state.get_memories(scope,['new'])['new'].active
    second=clear(c,w,request='another-clear').json();assert second['records']==1
    assert clear(c,w,request='empty-clear').json()['records']==0


@pytest.mark.parametrize('activity',['worker','chat'])
def test_busy_workspace_rejected_atomically_without_cancellation(env,activity):
    p,c,o,w,s=env;scope=p.store.scope(o,w,'textual');publish(p,scope,raw('keep','Keep while busy',s))
    if activity=='worker':
        p.store.enqueue(o,w,'index',{},['busy-test']);job=p.store.claim('index','busy-worker');assert job
    else:
        with p.state.transaction() as db:
            db.execute("INSERT INTO cs_chat(id,owner,workspace,session,request_id,digest,question,media,status,created_at,updated_at) VALUES ('busy',?,?,?,'busy','digest','question','[]','processing',1,1)",(o,w,s))
    response=clear(c,w);assert response.status_code==409 and response.json()['error']['code']=='workspace_busy'
    assert p.state.get_memories(scope,['keep'])['keep'].active
    assert not p.store.rows('SELECT * FROM cs_memory_clears')
    if activity=='worker':assert p.store.rows('SELECT status FROM cs_jobs WHERE id=?',(job['id'],))[0]['status']=='running'


def test_partial_clear_fences_writes_and_resumes_frozen_batches(env,monkeypatch):
    p,c,o,w,s=env;scope=p.store.scope(o,w,'textual')
    publish(p,scope,*[raw('source-'+str(i).zfill(3),'Test source '+str(i),s) for i in range(205)])
    original=p.coordinator.apply;calls=[]
    def fail_second(*args,**kwargs):
        calls.append(args[1])
        if len(calls)==2:raise RuntimeError('Explicit second-batch interruption injection')
        return original(*args,**kwargs)
    monkeypatch.setattr(p.coordinator,'apply',fail_second)
    assert clear(c,w).status_code==503
    assert p.store.rows("SELECT COUNT(*) n FROM memories WHERE active=1 AND scope=?",(scope.key,))[0]['n']==5
    payload={'workspace_id':w,'session_id':s,'request_id':'blocked-new-input','messages':[{'content':'Must not enter during recovery'}]}
    assert c.post('/api/memories',json=payload).status_code==409
    assert c.post('/api/chat',json={'workspace_id':w,'session_id':s,'request_id':'blocked-chat','content':'No generation during clearing'}).status_code==409
    assert not p.store.rows('SELECT 1 FROM cs_ingestions WHERE request_id=?',(payload['request_id'],))
    assert p.store.claim('index','during-clear') is None
    monkeypatch.setattr(p.coordinator,'apply',original)
    # A reopened modal can resume even if its original request ID was lost.
    retry=clear(c,w,request='browser-reopened');assert retry.status_code==200,retry.text
    assert retry.json()['records']==205
    assert p.store.rows("SELECT COUNT(*) n FROM memories WHERE active=1 AND scope=?",(scope.key,))[0]['n']==0
    assert c.post('/api/memories',json=payload).status_code==202


def test_duplicate_clear_requests_do_not_run_in_parallel(env,monkeypatch):
    p,c,o,w,s=env;scope=p.store.scope(o,w,'textual');publish(p,scope,raw('one','One source',s))
    entered=Event();release=Event();original=p.coordinator.apply
    def paused(*args,**kwargs):
        entered.set();assert release.wait(5);return original(*args,**kwargs)
    monkeypatch.setattr(p.coordinator,'apply',paused)
    with ThreadPoolExecutor(max_workers=1) as pool:
        first=pool.submit(p.memory_clearing.clear,o,w,'same-request')
        try:
            assert entered.wait(5)
            with pytest.raises(AppError) as error:p.memory_clearing.clear(o,w,'same-request')
            assert error.value.code=='memory_clear_busy'
        finally:release.set()
        assert first.result()['records']==1


def test_clear_between_retry_read_and_write_cannot_resurrect_cancelled_job(env,monkeypatch):
    p,c,o,w,s=env
    key=p.store.enqueue(o,w,'index',{},['retry-race'])
    with p.state.transaction() as db:db.execute("UPDATE cs_jobs SET status='failed' WHERE id=?",(key,))
    original=p.store.rows;interleaved=[]
    def clear_after_old_read(sql,params=()):
        rows=original(sql,params)
        if sql=='SELECT * FROM cs_jobs WHERE id=? AND owner=?' and not interleaved:
            interleaved.append(True);p.memory_clearing.clear(o,w,'race-clear')
        return rows
    monkeypatch.setattr(p.store,'rows',clear_after_old_read)
    with pytest.raises(AppError):p.store.retry(o,key)
    assert original('SELECT status FROM cs_jobs WHERE id=?',(key,))[0]['status']=='cancelled'


def test_restart_recovery_after_canonical_commit(tmp_path,monkeypatch):
    monkeypatch.setenv('DEEPSEEK_API_KEY','unit-only-not-a-real-key')
    settings=Settings(data_dir=tmp_path,sparse_url='',allow_retrieval_degraded=True)
    p=Platform(settings)
    with TestClient(create_app(platform=p,start_workers=False)) as c:
        identity=login_test(c,p);o=identity['user']['id'];w=identity['workspaces'][0]['id'];s=p.store.create_session(o,w,'textual','Kept conversation')['id'];scope=p.store.scope(o,w,'textual')
        publish(p,scope,raw('one','Durable clear source',s))
        def fault(point):
            if point=='after_T2_before_http_return':raise RuntimeError('Explicit crash-after-commit injection')
        p.coordinator.fault=fault
        assert clear(c,w).status_code==503
    reopened=Platform(settings)
    try:
        reopened.store.recover();reopened.coordinator.recover();reopened.memory_clearing.recover()
        assert reopened.store.rows('SELECT status FROM cs_memory_clears')[0]['status']=='completed'
        assert reopened.state.get_memories(scope,['one'])['one'].lifecycle=='deleted'
        assert reopened.store.session(o,w,s)['title']=='Kept conversation'
        assert reopened.store.claim('index','recovered-cleanup')['payload']['memory_clear_id']
    finally:reopened.close()


def test_repository_lookup_cleared_and_explicit_rescan_rebuilds_with_new_version(env,tmp_path):
    p,c,o,w,s=env;root=tmp_path/'repository';root.mkdir();subprocess.run(['git','init','-q',str(root)],check=True)
    (root/'sample.py').write_text('def add(a, b):\n    return a + b\n')
    p.settings.repository_roots=str(root)
    registered=p.repositories.register(o,w,str(root),'Test source','scan-before-clear');repo=registered['id']
    job=p.store.claim('repository','scan-before');p.store.finish(job,p.do_repository(job))
    scope=p.store.scope(o,w,'code',repo);before=p.repositories.files(o,w,repo,0,20)['items'][0]
    symbols=p.repositories.symbols(o,w,repo,0,20)['items'];assert symbols
    # No live index worker was started in this isolated fixture.
    assert clear(c,w).status_code==200
    assert not p.repositories.symbols(o,w,repo,0,20)['items']
    assert not p.repositories.files(o,w,repo,0,20)['items']
    assert p.store.repository(o,w,repo)['status']=='cleared'
    assert all(not m.active for m in p.state.get_memories(scope,[r['uid'] for r in symbols]).values())
    p.repositories.rescan(o,w,repo,'scan-after-clear');job=p.store.claim('repository','scan-after');p.store.finish(job,p.do_repository(job))
    assert p.repositories.files(o,w,repo,0,20)['items'][0]['version']==before['version']+1
    assert p.repositories.symbols(o,w,repo,0,20)['items']


def test_remote_same_commit_after_clear_rebuilds_from_verified_cache(env,monkeypatch):
    from tests.chinasoft.test_v32 import fixture_provider,scan
    from mandol_competition.chinasoft.remote_git import PublicHTTP
    p,c,o,w,s=env
    fixture_provider(monkeypatch,p,{'src/a.py':b'def alpha():\n    return 1\n','src/use.py':b'from src.a import alpha\n'})
    result=c.post('/api/repositories/remote',json={'workspace_id':w,'url':'https://github.com/example/fixture','request_id':'remote-before-clear'});assert result.status_code==202
    repo=result.json()['id'];before,_=scan(p);assert before['changed_files']==2
    assert clear(c,w).status_code==200
    # The checked blob cache may be reused, but both files must be reparsed and
    # republished. This fixture forbids downloading their bodies again.
    def no_download(*args,**kwargs):raise AssertionError('Expected SHA-verified cached blob reuse')
    monkeypatch.setattr(PublicHTTP,'get',no_download)
    p.repositories.rescan(o,w,repo,'remote-after-clear');after,_=scan(p)
    assert after['changed_files']==2 and not after.get('reused_commit')
    files=p.repositories.files(o,w,repo,0,20)['items'];assert len(files)==2
    assert all(f['version']==2 for f in files)
    assert p.repositories.symbols(o,w,repo,0,20)['items']
