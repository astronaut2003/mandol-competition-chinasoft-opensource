"""Actual SQL/API/publication regression; HTTP model failures injected explicitly.

Structured fixtures below are test data, not purported live model extractions.
Live model ablations are scripts/research/chinasoft_v4_eval.py.
"""
from dataclasses import replace
import json
import pytest
from fastapi.testclient import TestClient
from mandol_substrate.state import CanonicalMemory, Edge, Mutation, Scope
from mandol_competition.chinasoft.api import create_app
from mandol_competition.chinasoft.config import Settings
from mandol_competition.chinasoft.service import Platform
from mandol_competition.chinasoft.models import ModelError
from mandol_competition.chinasoft.query_planner import MemoryQueryPlanner, POLICIES, pack_context, absolute_range
from tests.chinasoft.auth_helpers import login_test, api_owner


@pytest.fixture
def p(tmp_path,monkeypatch):
    platform=Platform(Settings(data_dir=tmp_path,allow_retrieval_degraded=True,sparse_url='',memory_query_strategy='typed_graph'))
    def offline(*args,**kwargs):raise ModelError('injected_offline','Explicit model outage injection')
    monkeypatch.setattr(platform.retrieval,'encode_query',offline)
    monkeypatch.setattr(platform.retrieval,'rerank',offline)
    yield platform
    platform.close()


def setup(p,owner='v4-owner'):
    w=p.store.create_workspace(owner,'V4 SQL integration')['id']
    s=p.store.create_session(owner,w,'textual','history',unified=True)['id']
    return owner,w,s,p.store.scope(owner,w,'textual')


def publish(p,scope,*memories,edges=()):
    key='test-'+str(p.state.scope_state(scope)['canonical_seq'])
    p.coordinator.apply(scope,key,key,Mutation(memories=tuple(memories),edges=tuple(edges)),{})


def raw(key,text,session):
    return CanonicalMemory(key,text,session_id=session,role='user',memory_type='observation')


def derived(key,text,source,kind='engineering_experience',**metadata):
    return CanonicalMemory(key,text,role='derived',memory_type=kind,memory_level='L1_Summary',
        metadata={'source_versions':{source.logical_uid:source.version},**metadata})


def test_two_tenants_two_sessions_type_and_source_filters(p):
    a,w,s,scope=setup(p);b,wb,sb,bscope=setup(p,'other-owner')
    s2=p.store.create_session(a,w,'textual','second',unified=True)['id']
    first=raw('a-raw','海港数据库唯一约束修复 evidence A',s);second=raw('second-raw','海港数据库唯一约束修复 second session',s2)
    publish(p,scope,first,second,derived('experience-a','海港数据库唯一约束修复经验',first))
    publish(p,bscope,raw('secret-b','海港数据库唯一约束修复 SECRET OTHER TENANT',sb))
    result=p.search(a,w,'海港数据库唯一约束修复',retrieval_scope='workspace',conversation_first=True,session_id=s,source_kind='derived')
    assert [i['id'] for i in result['items']]==['experience-a']
    assert result['items'][0]['evidence_sources'][0]['id']=='a-raw'
    assert not any('SECRET' in json.dumps(i) or 'second session' in json.dumps(i) for i in result['items'])
    assert result['route']['trace']['extra_llm_calls']==0
    assert not result['route']['trace']['retrieval_complete']
    assert result['warnings']
    with pytest.raises(Exception) as exc:p.search(a,wb,'SECRET',strategy='C8_combined')
    assert exc.value.status==404


def test_source_revision_during_rerank_excludes_derived_and_changed_candidate(p,monkeypatch):
    a,w,s,scope=setup(p);source=raw('raw','海港故障恢复原始证据',s)
    publish(p,scope,source,derived('exp','海港故障恢复工程经验',source))
    def mutate(query,documents):
        publish(p,scope,replace(source,version=2,text_content='已经更正：原修复没有完成'))
        return list(enumerate([1.0]*len(documents)))
    monkeypatch.setattr(p.retrieval,'rerank',mutate)
    result=p.search(a,w,'海港故障恢复',retrieval_scope='workspace',conversation_first=True,strategy='C6_evidence')
    assert result['items']==[]
    assert result['route']['trace']['invalid_excluded']>=2
    assert p.state.get_memories(scope,['raw'])['raw'].version==2


def test_source_delete_and_excluded_current_ingestion_cannot_reenter_via_graph(p):
    a,w,s,scope=setup(p);source=raw('raw','陈青是海港数据库工程负责人',s)
    entity=derived('entity','陈青',source,'entity',hlm_item={'name':'陈青'})
    relation=derived('rel','陈青负责海港数据库',source,'relation')
    publish(p,scope,source,entity,relation,edges=(Edge('role','rel','entity','RELATION_SUBJECT'),))
    excluded=p.search(a,w,'陈青负责什么数据库',retrieval_scope='workspace',conversation_first=True,strategy='C8_combined',exclude_sources=['raw'])
    assert excluded['items']==[]
    p.coordinator.apply(scope,'delete','delete',Mutation(delete_uids=('raw',)),{})
    assert not p.search(a,w,'陈青负责什么数据库',retrieval_scope='workspace',conversation_first=True,strategy='typed_graph')['items']


def test_graph_uses_stored_edges_and_rejects_cross_scope_alias(p):
    a,w,s,scope=setup(p);b,wb,sb,bscope=setup(p,'b-owner')
    source=raw('raw','陈青负责测试，林老师主管星港实验室。',s)
    entity=derived('entity','陈青',source,'entity',hlm_item={'name':'陈青'})
    relation=derived('rel','星港实验室主管是林老师',source,'relation')
    foreign=raw('foreign','SECRET foreign resource',sb);publish(p,bscope,foreign)
    alias=CanonicalMemory('bad-alias','reference',role='derived',memory_type='canonical_reference',metadata={'hlm_reference':{'scope_key':bscope.key,'uid':'foreign','version':1,'text_hash':'invalid'}})
    publish(p,scope,source,entity,relation,alias,edges=(Edge('proof','rel','entity','RELATION_SUBJECT'),Edge('attack','entity','bad-alias','PARTICIPATES_IN')))
    result=p.search(a,w,'陈青所在实验室主任是谁',retrieval_scope='workspace',conversation_first=True,strategy='C3_graph')
    trace=result['route']['trace'];assert trace['lanes']['graph']['candidates']>=1
    assert any(path['to']=='rel' for path in trace['graph_paths'])
    assert all(path['to']!='foreign' for path in trace['graph_paths'])
    assert 'SECRET' not in json.dumps(result)
    # Simulate a canonical edge whose publication is still pending. Existing
    # valid nodes do not authorize following an unpublished relationship.
    frontier=p.state.scope_state(scope)['published_seq']
    with p.state.transaction() as db:db.execute('UPDATE graph_edges SET commit_seq=? WHERE scope=?',(frontier+1,scope.key))
    pending=p.search(a,w,'陈青所在实验室主任是谁',retrieval_scope='workspace',conversation_first=True,strategy='C3_graph')
    assert pending['route']['trace']['lanes']['graph']['candidates']==0
    assert pending['route']['trace']['graph_paths']==[]


def test_explicit_event_time_and_unknown_time_never_use_created_timestamp(p):
    a,w,s,scope=setup(p);source=raw('raw','9月在杭州，10月南京，另一个事件时间未知。',s)
    memories=[source]
    for name,event,precision in [('old','2026-09-12','DAY'),('new','2026-10-03','DAY'),('unknown',None,'UNKNOWN')]:
        memories.append(derived(name,'项目演示 '+name,source,'episode',hlm_item={'event_time':event,'time_precision':precision}))
    publish(p,scope,*memories)
    r=p.search(a,w,'2026年10月项目演示发生什么',retrieval_scope='workspace',conversation_first=True,strategy='C2_lifecycle')
    ids={m['id'] for m in r['items']};assert 'old' not in ids and {'new','unknown'}<=ids
    assert next(i for i in r['items'] if i['id']=='unknown')['temporal_status']=='unknown_event_time_kept'


def test_fast_code_lookup_preserves_filename_lines_and_falls_back_on_ambiguity(p,monkeypatch):
    a,w,s,scope=setup(p);source=raw('raw','def stable_key(owner):\n    return owner\n',s)
    symbol=derived('symbol','def stable_key(owner): return owner',source,'code_symbol',memory_family='code',source_locator={'path':'keys.py','start_line':1,'end_line':2})
    symbol=replace(symbol,raw_data={'name':'stable_key','qualified_name':'stable_key'})
    publish(p,scope,source,symbol)
    def forbidden(*a,**k):raise AssertionError('Exact unique lookup must not encode')
    monkeypatch.setattr(p.retrieval,'encode_query',forbidden)
    r=p.search(a,w,'函数 stable_key 定义在哪个文件哪几行？',retrieval_scope='workspace',conversation_first=True,strategy='C7_cost')
    assert r['route']['trace']['fast_exact'] and r['items'][0]['source_locator']['path']=='keys.py'
    second=replace(symbol,logical_uid='other-symbol',metadata={**symbol.metadata,'source_locator':{'path':'other.py','start_line':4,'end_line':5}})
    publish(p,scope,second)
    with pytest.raises(AssertionError):p.search(a,w,'函数 stable_key 定义在哪？',retrieval_scope='workspace',conversation_first=True,strategy='C7_cost')


def test_multi_label_plan_and_bounded_context_no_token_llm_or_scope_widening():
    plan=MemoryQueryPlanner().plan('之前会议白板的OCR中数据库修复方案是什么？',policy=POLICIES['C8_combined'],scope_filters={'workspace_id':'authorized-only','conversation_id':'one'},available_types={'observation','visual_ocr','visual_memory','engineering_experience','episode'})
    assert {'ocr','engineering','visual','episode'}<=set(plan.query_intents)
    assert sum(l.candidate_budget for l in plan.selected_lanes)==16
    assert plan.scope_filters['workspace_id']=='authorized-only' and plan.fallback_policy=='bounded_raw_hybrid'
    assert absolute_range('2026年10月')==('2026-10-01','2026-11-01')
    assert absolute_range('昨天') is None


def test_catalog_budget_counts_filtered_out_metadata_across_scopes(p,monkeypatch):
    from mandol_competition.chinasoft import typed_retrieval
    a,w,s,scope=setup(p);other=p.store.scope(a,w,'code')
    publish(p,scope,raw('one','first searchable record',s),raw('two','second searchable record',s))
    publish(p,other,raw('three','third searchable record',s),raw('four','fourth searchable record',s))
    # Lower only the read budget, using actual SQL/publication and authorized
    # scopes. A result-type filter must not hide catalog reads from its budget.
    monkeypatch.setattr(typed_retrieval,'MAX_SCOPE_CATALOG',2)
    monkeypatch.setattr(typed_retrieval,'MAX_TOTAL_CATALOG',3)
    result=p.search(a,w,'searchable record',retrieval_scope='workspace',conversation_first=True,memory_type='code_symbol',strategy='typed')
    assert result['route']['planner_fallback']=='typed_metadata_budget_exceeded'
    assert result['items']==[]


def test_source_expansion_budget_is_shared_by_all_derived_hits(p):
    a,w,s,scope=setup(p)
    sources=[raw('source-'+str(i),'Original evidence '+str(i),s) for i in range(34)]
    first=derived('experience-one','共享来源预算检索实例',sources[0])
    second=derived('experience-two','共享来源预算检索实例',sources[17])
    first=replace(first,metadata={'source_versions':{m.logical_uid:1 for m in sources[:17]}})
    second=replace(second,metadata={'source_versions':{m.logical_uid:1 for m in sources[17:]}})
    publish(p,scope,*sources,first,second)
    result=p.search(a,w,'共享来源预算检索实例',retrieval_scope='workspace',conversation_first=True,source_kind='derived',strategy='C6_evidence_v3')
    assert len(result['items'])==2
    unique={m['id'] for hit in result['items'] for m in hit['evidence_sources']}
    assert len(unique)==24 and result['route']['trace']['expanded_source_unique']==24


def test_chat_tool_route_cannot_select_missing_current_attachments(p,monkeypatch):
    from mandol_competition.chinasoft.conversation import plan_turn
    # An explicitly injected classifier label tests structural validation. The
    # real failure and real-model rerun are preserved separately in live*.json.
    def classified(*args,**kwargs):
        return {'content':json.dumps({'intent':'attachments','retrieve':False,'use_attachments':True,'query':'old control panel'}),'response_model':'test-classifier'}
    monkeypatch.setattr(p.generation,'complete',classified)
    result=plan_turn(p,'请解释此前上传的控制台截图',[],[],None)
    assert result['retrieve'] and result['intent']=='personal_history'
    assert not result['use_attachments']
    assert result['route_validation']=='current_attachments_unavailable_retrieve_history'
    # A real current image is a different capability and remains available
    # without a mandatory memory retrieval.
    current=plan_turn(p,'请解释此前上传的控制台截图',[],[],None,has_images=True)
    assert not current['retrieve'] and current['intent']=='attachments'


def test_evidence_dedup_and_small_context_keeps_valid_source_identity():
    source={'id':'r','version':2,'expected_version':2,'role':'user','active':True,'text':'user statement','scope_key':'s'}
    m={'id':'d','version':1,'source_kind':'derived','memory_type':'engineering_experience','memory_family':'code','repository_id':'','role':'derived','verification':'SOURCE_CITED','text':'summary','evidence_sources':[source]}
    blocks,citations,trace=pack_context([m,{**m,'id':'second'}],expand=True,budget=2000)
    assert len(blocks)==len(citations)==2 and trace['source_ids']==['r'] and trace['duplicate_sources_skipped']>0
    assert blocks[1]['original_sources']==[] and blocks[1]['source_refs'][0]['id']=='r'
    assert pack_context([m],expand=True,budget=5)[0]==[]


def test_raw_caption_does_not_mask_visual_description_and_ocr():
    raw={'id':'image','version':1,'scope_key':'s','memory_type':'image_text','memory_family':'multimodal','repository_id':'',
         'role':'user','source_kind':'raw','verification':'ORIGINAL','text':'My status board','sources':[]}
    support={'id':'image','version':1,'expected_version':1,'scope_key':'s','text':raw['text'],'role':'user','active':True,'media_ids':['real-image-id']}
    visual={**raw,'id':'description','role':'derived','source_kind':'derived','memory_type':'visual_description','text':'API SERVER ONLINE','evidence_sources':[support]}
    ocr={**visual,'id':'ocr','memory_type':'ocr_text','text':'Room A-203'}
    blocks,citations,_=pack_context([raw,visual,ocr],expand=True)
    assert len(blocks)==3 and 'API SERVER ONLINE' in json.dumps(blocks) and 'Room A-203' in json.dumps(blocks)
    assert blocks[1]['source_refs'][0]['media_ids']==['real-image-id']
    # The old research candidate is retained to reproduce the rejected result.
    assert len(pack_context([raw,visual,ocr],expand=True,dedup_mode='source-family-v1')[0])==1


def test_engineering_history_cannot_impersonate_current_source_in_context():
    record={'id':'reported','version':1,'source_kind':'raw','memory_type':'conversation_message','memory_family':'conversation',
        'repository_id':'','role':'user','verification':'ORIGINAL','text':'我修改过 ledger.py 并运行测试。','metadata':{'routes':['engineering']}}
    block=pack_context([record],with_basis=True)[0][0]
    assert block['evidence_basis'].startswith('ENGINEERING_REPORT:') and 'NOT proof' in block['evidence_basis']
    static={**record,'id':'source','memory_type':'code_symbol','source_kind':'derived','metadata':{'source_locator':{'path':'ledger.py'}}}
    assert pack_context([static],with_basis=True)[0][0]['evidence_basis'].startswith('STATIC_SOURCE:')


def test_api_authorization_applies_to_planner_and_trace(p):
    with TestClient(create_app(platform=p,start_workers=False)) as a,TestClient(create_app(platform=p,start_workers=False)) as b:
        wa=login_test(a,p)['workspaces'][0]['id'];wb=login_test(b,p)['workspaces'][0]['id']
        owner=api_owner(a,p);s=p.store.create_session(owner,wa,'textual','A',unified=True)['id'];scope=p.store.scope(owner,wa,'textual')
        publish(p,scope,raw('private','PRIVATE SQL SOURCE',s))
        params={'workspace_id':wa,'q':'PRIVATE','retrieval_scope':'conversations'}
        assert b.get('/api/memories/search',params=params).status_code==404
        response=a.get('/api/memories/search',params=params);assert response.status_code==200
        assert response.json()['route']['trace']['plan']['scope_filters']['workspace_id']==wa


def test_planner_does_not_change_vector_profile_or_memory_versions(p):
    a,w,s,scope=setup(p);publish(p,scope,raw('r','既有事实',s))
    profile=p.vector_profile.hash;before=p.state._rows('SELECT * FROM memories')
    p.search(a,w,'既有事实',retrieval_scope='workspace',conversation_first=True,strategy='C8_combined')
    assert p.vector_profile.hash==profile and p.state._rows('SELECT * FROM memories')==before
