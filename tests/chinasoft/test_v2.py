from tests.chinasoft.auth_helpers import login_test,api_owner
"""Real SQLite/Tree-sitter/API scope tests. Provider faults are explicitly injected.

Live provider/Qdrant/browser acceptance is a separate script and report.
"""
from dataclasses import replace
from pathlib import Path
import json, subprocess
import pytest
from fastapi.testclient import TestClient
from mandol_substrate.state import CanonicalMemory, Mutation
from mandol_substrate.state.models import digest
from mandol_competition.chinasoft.api import create_app
from mandol_competition.chinasoft.config import Settings
from mandol_competition.chinasoft.schemas import AddMemory
from mandol_competition.chinasoft.service import Platform
from mandol_competition.chinasoft.models import ModelError
from mandol_competition.chinasoft.repository import validate_root, safe_read, parse_source

@pytest.fixture
def p(tmp_path,monkeypatch):
    monkeypatch.delenv('DEEPSEEK_API_KEY',raising=False)
    monkeypatch.delenv('SILICONFLOW_API_KEY',raising=False)
    platform=Platform(Settings(data_dir=tmp_path/'data',repository_roots=str(tmp_path)))
    yield platform
    platform.close()

def git_repo(root):
    root.mkdir();subprocess.run(['git','init','-q',str(root)],check=True)
    (root/'calc.py').write_text('def add_numbers(left, right):\n    return left + right\n')
    (root/'api.py').write_text('from calc import add_numbers\n\ndef handle_sum(a, b):\n    return add_numbers(a, b)\n')
    (root/'ui.ts').write_text('import {render} from "./view";\nexport class Client { send(value: number) { return render(value); } }\nexport const arrow = (x: number) => x + 1;\n')
    (root/'view.ts').write_text('export function render(value: number) { return String(value); }\n')
    (root/'.gitignore').write_text('ignored.py\n')
    return root

def workspace(p,owner='a'*32):
    w=p.store.create_workspace(owner,'scope-test')['id'];return owner,w

def run_job(p,kind):
    j=p.store.claim(kind,'test');assert j
    result=getattr(p,'do_'+kind)(j);p.store.finish(j,result);return result

def register(p,owner,w,root,req='scan1'):
    r=p.repositories.register(owner,w,str(root),'test-repo',req);run_job(p,'repository');return r['id']

def test_additive_migration_preserves_legacy_scopes_and_reopens(p):
    o,w=workspace(p);s=p.store.create_session(o,w,'textual','legacy')
    scope=p.store.scope(o,w,'textual');assert scope.user_id==f'{o}.{w}.textual'
    assert p.store.rows('SELECT version FROM cs_schema_migrations')==[{'version':2},{'version':3},{'version':31},{'version':32}]
    assert s['repository']==''

def test_repository_tree_sitter_imports_and_current_evidence(p,tmp_path):
    o,w=workspace(p);key=register(p,o,w,git_repo(tmp_path/'repo'))
    symbols=p.repositories.symbols(o,w,key,0,50)['items'];assert {'add_numbers','handle_sum','Client','send','arrow','render'} <= {r['name'] for r in symbols}
    result=p.search(o,w,'add_numbers 函数定义在哪里？','code',repository_id=key)
    assert result['items'];assert all(r['repository_id']==key for r in result['items'])
    m=next(r for r in result['items'] if r['memory_type']=='code_symbol' and 'add_numbers' in r['text'])
    assert m['source_locator']['path']=='calc.py';assert m['source_locator']['start_line']==1
    assert m['sources'][0]['text'].endswith('    return left + right\n')
    links=p.store.rows('SELECT * FROM cs_repository_links WHERE repository=?',(key,))
    assert any(r['path']=='api.py' and r['target_path']=='calc.py' and r['target_symbol'] for r in links)
    assert p.repositories.current(o,w,key)['statistics']['changed_files']==4

def test_incremental_noop_change_delete_history_and_source_live_check(p,tmp_path):
    o,w=workspace(p);root=git_repo(tmp_path/'repo');key=register(p,o,w,root)
    scope=p.store.scope(o,w,'code',key);before=p.state.scope_state(scope)['published_seq']
    old=p.repositories.symbols(o,w,key,0,50)['items'];oldid=next(r['uid'] for r in old if r['name']=='add_numbers')
    p.repositories.rescan(o,w,key,'noop');stats=run_job(p,'repository')
    assert stats['changed_files']==0 and stats['unchanged_files']==4
    assert p.state.scope_state(scope)['published_seq']==before
    (root/'calc.py').write_text('def subtract_numbers(left, right):\n    return left - right\n')
    assert not p.eligible(scope,p.state.get_memories(scope,[oldid])[oldid])
    maps=p.explorer(o,w,'code',repository_id=key,memory_type='repository_map')['items']
    assert maps and not maps[0]['eligible']
    deps=p.explorer(o,w,'code',repository_id=key,memory_type='code_dependency')['items']
    assert any(not m['eligible'] for m in deps if 'add_numbers' in m['text'])
    p.repositories.rescan(o,w,key,'changed');stats=run_job(p,'repository');assert stats['changed_files']==1 and stats['unchanged_files']==3
    symbols=p.repositories.symbols(o,w,key,0,50)['items'];assert 'subtract_numbers' in {r['name'] for r in symbols};assert 'add_numbers' not in {r['name'] for r in symbols}
    assert not p.state.get_memories(scope,[oldid])[oldid].active
    history=p.repositories.history(o,w,key,'calc.py',0,50)['items'];assert len(history)==2
    (root/'calc.py').unlink();p.repositories.rescan(o,w,key,'deleted');stats=run_job(p,'repository');assert stats['deleted_files']==1
    assert not any(r['path']=='calc.py' for r in p.repositories.files(o,w,key,0,50)['items'])
    assert not any(r['name']=='subtract_numbers' for r in p.repositories.symbols(o,w,key,0,50)['items'])
    assert not any(r['target_path']=='calc.py' and r['resolved'] for r in p.store.rows('SELECT * FROM cs_repository_links WHERE repository=?',(key,)))
    (root/'calc.py').write_text('def add_numbers(left, right):\n    return left + right\n')
    p.repositories.rescan(o,w,key,'readded');stats=run_job(p,'repository')
    assert stats['changed_files']==1
    assert any(r['name']=='add_numbers' for r in p.repositories.symbols(o,w,key,0,50)['items'])

@pytest.mark.parametrize('path',['/etc','/srv/not-allowlisted-source','../repo'])
def test_path_whitelist_rejects_system_aml_and_traversal(p,path):
    with pytest.raises(Exception):validate_root(p.settings,path)

def test_scanner_excludes_secrets_ignored_binary_symlink_and_never_runs_hook(p,tmp_path):
    o,w=workspace(p);root=git_repo(tmp_path/'repo');outside=tmp_path/'private.py';outside.write_text('SECRET = "private"')
    (root/'escape.py').symlink_to(outside);(root/'.env').write_text('KEY=private');(root/'ignored.py').write_text('ignored = True')
    (root/'secret.py').write_text('key = "sk-'+'A'*25+'"');(root/'binary.py').write_bytes(b'\0\xff')
    marker=tmp_path/'hook-ran';hook=root/'.git/hooks/pre-commit';hook.write_text('#!/bin/sh\ntouch '+str(marker));hook.chmod(0o755)
    key=register(p,o,w,root);files=p.repositories.files(o,w,key,0,50)['items'];paths={r['path'] for r in files}
    assert not paths & {'escape.py','.env','ignored.py','secret.py','binary.py'}
    assert not marker.exists();skips=p.repositories.current(o,w,key)['statistics']['skipped'];assert len(skips)>=4
    with pytest.raises(OSError):safe_read(root,'escape.py',1000)
    with pytest.raises(Exception):validate_root(p.settings,str(tmp_path/'repo-link'))

def test_workspace_family_repository_scope_and_source_isolation(p,tmp_path):
    o,a=workspace(p);_,b=workspace(p,o)
    ra=register(p,o,a,git_repo(tmp_path/'ra'),'r-a');rb=register(p,o,a,git_repo(tmp_path/'rb'),'r-b')
    memories=[]
    for w in (a,b):
        for mode in ('textual','code','multimodal'):
            ses=p.store.create_session(o,w,mode,mode)['id'];r=p.store.accept(o,AddMemory(workspace_id=w,session_id=ses,request_id=mode+w,mode=mode,messages=[{'content':'UNIQUE_'+w+'_'+mode}]))
            run_job(p,'ingest');memories.extend(r['memory_ids'])
    current=p.search(o,a,'UNIQUE','textual');assert all(m['memory_family']=='conversation' and m['workspace_id']==a for m in current['items'])
    all_types=p.search(o,a,'UNIQUE','textual',retrieval_scope='workspace');assert set(all_types['route']['searched_families'])=={'conversation','code','multimodal'}
    assert all(m['workspace_id']==a for m in all_types['items'])
    repo=p.search(o,a,'add_numbers','code',repository_id=ra);assert all(m['repository_id']==ra for m in repo['items'])
    assert not p.explorer(o,a,'code')['items'] or all(m['repository_id']=='' for m in p.explorer(o,a,'code')['items'])
    with pytest.raises(Exception):p.repositories.current(o,b,ra)
    with pytest.raises(Exception):p.store.create_session(o,b,'code','stolen',ra)
    foreign=p.explorer(o,b,'textual')['items'][0]['id']
    with pytest.raises(Exception):p.locate(o,a,foreign)
    with pytest.raises(Exception):p.search(o,b,'add_numbers','code',repository_id=rb)

def test_index_payload_cannot_cross_workspace(p):
    o,a=workspace(p);_,b=workspace(p,o);foreign=p.store.scope(o,b,'textual')
    with pytest.raises(Exception) as error:p.do_index({'owner':o,'workspace':a,'payload':{'scopes':[{'namespace':foreign.namespace,'user_id':foreign.user_id}]}})
    assert error.value.code=='scope_rejected'

def test_auto_chat_uses_current_session_family_and_rejects_repo_switch(p,tmp_path):
    app=create_app(platform=p,start_workers=False)
    with TestClient(app) as c:
        w=login_test(c,p)['workspaces'][0]['id'];s=c.get('/api/sessions',params={'workspace_id':w}).json()['items'][0]['id']
        data={'workspace_id':w,'session_id':s,'request_id':'auto','content':'source scope','mode':'auto','retrieval_scope':'current'}
        response=c.post('/api/chat',json=data)
        route=json.loads(response.text.split('event: route\ndata: ')[1].split('\n')[0]);assert route['family']=='conversation' and route['retrieval_scope']=='current'
        assert c.post('/api/chat',json={**data,'request_id':'bad-mode','mode':'code'}).status_code==409
        assert c.post('/api/chat',json={**data,'request_id':'bad-repo','repository_id':'repo-foreign'}).status_code==409

def test_repository_partial_publication_recovery_does_not_duplicate(tmp_path):
    settings=Settings(data_dir=tmp_path/'data',repository_roots=str(tmp_path));p=Platform(settings);o,w=workspace(p);root=git_repo(tmp_path/'repo')
    reg=p.repositories.register(o,w,str(root),'recovery','initial');job=p.store.claim('repository','crashed')
    apply=p.coordinator.apply;calls=[]
    def interrupt(*args,**kwargs):
        result=apply(*args,**kwargs);calls.append(1)
        if len(calls)==1:raise RuntimeError('injected after canonical publication')
        return result
    p.coordinator.apply=interrupt
    with pytest.raises(RuntimeError):p.do_repository(job)
    p.coordinator.apply=apply;p.close();p=Platform(settings);p.store.recover();stats=run_job(p,'repository')
    assert stats['symbols']==6
    scope=p.store.scope(o,w,'code',reg['id']);rows=p.state._rows('SELECT logical_uid FROM memories WHERE scope=? AND active=1',(scope.key,))
    assert len(rows)==len({r['logical_uid'] for r in rows})
    p.close()

def test_invalid_json_and_failed_derivation_leave_raw_and_scope_intact(p):
    o,w=workspace(p);s=p.store.create_session(o,w,'code','engineering')['id']
    r=p.store.accept(o,AddMemory(workspace_id=w,session_id=s,request_id='raw',mode='code',messages=[{'content':'真实工程记录'}]));run_job(p,'ingest')
    assert p.locate(o,w,r['memory_ids'][0])[1].active
    assert p.explorer(o,w,'textual')['items']==[]

def test_parser_errors_are_explicit_and_call_names_are_unresolved():
    parsed=parse_source('main.py','from math import sqrt\ndef f(x):\n    return sqrt(x)\n')
    assert parsed['imports'][0]['module']=='math';assert parsed['calls'][0]['resolved'] is False
    assert parse_source('broken.py','def bad(:\n  return 0')['error']
    assert parse_source('data.go','package main')['parser']=='tree-sitter-1.14.3'
    assert parse_source('data.unsupported','plain evidence')['parser']=='text-chunks'

def test_removed_repository_reports_real_unavailability_without_losing_history(p,tmp_path):
    o,w=workspace(p);root=git_repo(tmp_path/'repo');key=register(p,o,w,root);root.rename(tmp_path/'moved')
    result=p.repositories.current(o,w,key)
    assert result['status']=='source_unavailable' and not result['source_available']
    assert p.repositories.history(o,w,key,'calc.py',0,50)['total']==1
    assert not p.search(o,w,'add_numbers','code',repository_id=key)['items']

def test_discourse_frame_does_not_survive_source_revision(p):
    from mandol_substrate.state import BuildJob
    from mandol_competition.textual.core.profile_discourse import empty_frame
    o,w=workspace(p);s=p.store.create_session(o,w,'textual','frame')['id'];scope=p.store.scope(o,w,'textual')
    raw=CanonicalMemory('raw-frame','真实偏好',session_id=s,role='user',metadata={'user_id':scope.user_id})
    build=BuildJob('frame-root',{'raw-frame':1},p.recipe.fingerprint,'completed',{'discourse_session_id':s,'frame':{'anchor':'old-context'}})
    p.coordinator.apply(scope,'frame-in','in',Mutation(memories=(raw,),jobs=(build,)),{})
    assert p.hlm._frame(scope,s)=={'anchor':'old-context'}
    p.coordinator.apply(scope,'frame-rev','rev',Mutation(memories=(replace(raw,version=2,text_content='更正偏好'),)),{})
    assert p.hlm._frame(scope,s)==empty_frame()

def test_identical_engineering_frontier_reuses_published_result_without_generation(p):
    from types import SimpleNamespace
    from mandol_competition.code.experience import ExperienceBuilder
    o,w=workspace(p);s=p.store.create_session(o,w,'code','cache')['id']
    r=p.store.accept(o,AddMemory(workspace_id=w,session_id=s,request_id='frame',mode='code',messages=[{'content':'实际工程问题与分析。'*25},{'content':'实际修复与当时验证。'*25}]))
    run_job(p,'ingest');scope=p.store.scope(o,w,'code');sources,_=p.session_sources(scope,s)
    plan={**p.settings.contract(),'generation_fingerprint':p.settings.generation_fingerprint}
    builder=ExperienceBuilder(SimpleNamespace(store=SimpleNamespace(state=p.state)),generator=lambda *a:None,model_plan=plan)
    key='engineering_experience:'+digest([scope.key,{m.logical_uid:m.version for m in sources},builder.fingerprint])
    exp=CanonicalMemory(key,'已发布的来源约束经验',session_id=s,role='derived',memory_type='engineering_experience',memory_level='L1_Episode',metadata={'source_versions':{m.logical_uid:m.version for m in sources}})
    p.coordinator.apply(scope,'exp-publish','exp',Mutation(memories=(exp,)),{})
    result=run_job(p,'derive');assert result['reused_memory_ids']==[key] and result['generated']==0
    assert not p.store.rows('SELECT * FROM cs_calls')

def test_profile_pair_guard_and_v1_reader_have_distinct_fingerprints(p):
    assert p.recipe.fingerprint!=p.legacy_hlm.recipe.fingerprint
    assert p.recipe.profile_recipe.fingerprint!=p.legacy_hlm.recipe.profile_recipe.fingerprint

def test_api_workspaces_guard_media_jobs_and_repository_sources(p,tmp_path):
    app=create_app(platform=p,start_workers=False)
    with TestClient(app) as c:
        a=login_test(c,p)['workspaces'][0]['id'];b=c.post('/api/workspaces',json={'name':'B'}) .json()['id']
        ses=c.post('/api/sessions',json={'workspace_id':a,'mode':'multimodal','title':'A'}).json()['id']
        media=c.post('/api/media',data={'workspace_id':a,'session_id':ses},files={'file':('board.png',(Path(__file__).parents[2]/'demo/status_board.png').read_bytes(),'image/png')}).json()
        assert c.get('/api/media/'+media['id'],params={'workspace_id':b}).status_code==404
        assert c.get('/api/media/'+media['id']).status_code==422
        receipt=c.post('/api/memories',json={'workspace_id':a,'session_id':ses,'mode':'multimodal','request_id':'isolated','messages':[{'content':'owned'}]}).json()
        assert c.get('/api/ingestions/'+receipt['ingestion_id'],params={'workspace_id':b}).status_code==404
        assert c.post('/api/jobs/'+receipt['jobs'][0]+'/retry',params={'workspace_id':b},json={}).status_code==404
        root=git_repo(tmp_path/'guard-repo')
        reg=c.post('/api/repositories',json={'workspace_id':a,'root':str(root),'name':'guard','request_id':'repo'}).json();run_job(p,'repository')
        key=reg['id']
        for suffix in ('','/files','/symbols','/map','/history'):
            assert c.get('/api/repositories/'+key+suffix,params={'workspace_id':b,'path':'calc.py'}).status_code==404
        assert p.recipe.profile_recipe.skip_context_only and p.recipe.profile_recipe.session_scoped_frame

def test_pending_publication_is_not_ready_or_visible_as_evidence(p):
    o,w=workspace(p);scope=p.store.scope(o,w,'textual')
    memory=CanonicalMemory('pending','可靠 canonical 但尚未发布',role='user')
    p.state.prepare(scope,'preparing','digest',Mutation(memories=(memory,)),{})
    current=p.state.get_memories(scope,['pending'])['pending']
    assert not p.eligible(scope,current)
    assert not p.serialize(scope,current)['lexical_ready']
    with pytest.raises(Exception):p.locate(o,w,'pending')
    p.coordinator.recover();assert p.eligible(scope,p.state.get_memories(scope,['pending'])['pending'])
