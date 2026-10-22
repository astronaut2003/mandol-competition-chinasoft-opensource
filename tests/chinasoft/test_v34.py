"""Real SQLite/publication/HTTP quality tests. Structured fixtures are not VLM output."""
from dataclasses import replace
import pytest
from mandol_substrate.state import CanonicalMemory,Mutation,Edge
from mandol_competition.chinasoft.visualization import MemoryViews
from mandol_competition.chinasoft.semantic_schema import overview
from mandol_competition.chinasoft.memory_quality import validity,group_records
from tests.chinasoft.test_v33 import env
from tests.chinasoft.test_v4_retrieval import publish,raw,derived
from tests.chinasoft.auth_helpers import login_test


def profile(uid,source,**changes):
    values={'subject':'user','attribute':'sport','value':'basketball','kind':'preference','lifetime':'stable','status':'current','condition':None,'scope':None,'time_qualifier':None,'valid_from':None,'valid_to':None,**changes}
    return replace(derived(uid,'我喜欢篮球。',source,'profile'),metadata={'source_versions':{source.logical_uid:source.version},'hlm_item':{'profile_item':values},'verification':'EXPLICIT_TEST_CANONICAL_FIXTURE'})


def test_changed_source_exits_current_graph_cards_schema_and_retrieval_eligibility(env):
    p,c,o,w,s=env;scope=p.store.scope(o,w,'textual');a=raw('source','I prefer basketball',s);b=derived('higher','Basketball preference',a)
    publish(p,scope,a,b)
    publish(p,scope,replace(a,version=2,text_content='Correction: I prefer swimming'))
    assert not p.eligible(scope,b) and validity(p,scope,b)['state']=='source_version_changed'
    v=MemoryViews(p)
    assert 'higher' not in {n['id'] for n in v.graph(o,w,s)['nodes']}
    audit=v.graph(o,w,s,state_view='history')
    assert {n['id'] for n in audit['nodes']}=={'source','higher'}
    assert audit['history_record_total']==1 and audit['context_nodes_included']==1
    params={'workspace_id':w,'session_id':s,'retrieval_scope':'conversation'}
    assert c.get('/api/memories',params=params).json()['total']==1
    assert c.get('/api/memories',params=dict(params,state_view='history')).json()['total']==1
    current=c.get('/api/memory-schema',params=params).json();old=c.get('/api/memory-schema',params=dict(params,state_view='history')).json()
    assert current['instance']['counted']==1 and current['scope_totals']['records']==2 and current['scope_totals']['current']==1 and current['scope_totals']['history']==1
    assert old['instance']['counted']==1
    fresh=replace(b,logical_uid='rebuilt',metadata={'source_versions':{'source':2}});publish(p,scope,fresh)
    assert {n['id'] for n in v.graph(o,w,s)['nodes']}=={'source','rebuilt'}


def test_current_pagination_counts_after_eligibility_not_before(env):
    p,c,o,w,s=env;scope=p.store.scope(o,w,'textual');a=raw('source','source',s)
    highs=[derived('high-'+str(i),'same old summary',a) for i in range(30)]
    publish(p,scope,a,*highs);publish(p,scope,replace(a,version=2,text_content='changed'))
    v=MemoryViews(p);records,total=v.select(o,w,s,offset=0,limit=1)
    assert total==1 and records[0][1].logical_uid=='source'
    assert v.select(o,w,s,offset=1,limit=1)[0]==[]


def test_three_sources_fold_profiles_only_in_view_and_members_preserve_all_uids(env):
    p,c,o,w,s=env;scope=p.store.scope(o,w,'textual');objects=[];sessions=[s]+[p.store.create_session(o,w,'textual','other',unified=True)['id'] for _ in range(2)]
    for i,session in enumerate(sessions):
        a=raw('raw-'+str(i),'我喜欢篮球。',session);objects.extend((a,profile('profile-'+str(i),a)))
    publish(p,scope,*objects);v=MemoryViews(p);g=v.graph(o,w,aggregate=True,relationship_view='semantic')
    assert len(g['nodes'])==1 and g['nodes'][0]['record_count']==3
    group=g['nodes'][0];members=v.members(o,w,group['id'])
    assert {m['id'] for m in members['items']}=={'profile-0','profile-1','profile-2'}
    assert {r['id'] for m in members['items'] for r in m['sources']}=={'raw-0','raw-1','raw-2'}
    expanded=v.graph(o,w,aggregate=True,relationship_view='semantic',expand_group=group['id'])
    assert {n['id'] for n in expanded['nodes']}=={'profile-0','profile-1','profile-2'}
    assert p.state._rows('SELECT COUNT(*) n FROM memories')[0]['n']==6


@pytest.mark.parametrize('changes',[{'subject':'third_party'},{'condition':'at work'},{'valid_from':'2026-10-01'},{'status':'historical'},{'lifetime':'temporary'},{'value':'swimming'}])
def test_different_subject_time_condition_value_or_state_never_fold(env,changes):
    p,c,o,w,s=env;scope=p.store.scope(o,w,'textual');a=raw('a','one',s);b=raw('b','two',s)
    publish(p,scope,a,b,profile('one',a),profile('two',b,**changes))
    groups,_=group_records(MemoryViews(p).select(o,w,s,state_view='all')[0],o,w)
    assert not any(len(g)>1 for g in groups.values())


def test_similar_text_different_types_and_entity_subject_ids_are_not_identity(env):
    p,c,o,w,s=env;scope=p.store.scope(o,w,'textual');a=raw('a','same statement',s)
    x=profile('profile',a);y=derived('episode','我喜欢篮球。',a,'episode')
    r1=replace(derived('r1','same relation text',a,'relation'),metadata={'source_versions':{'a':1},'hlm_item':{'subject_id':'E0','object_id':'E2'}})
    r2=replace(r1,logical_uid='r2',metadata={**r1.metadata,'hlm_item':{'subject_id':'E1','object_id':'E2'}})
    # Text normalization is useful for diagnostics, not identity equivalence.
    u1=replace(derived('unicode-1','same label',a,'entity'),metadata={'source_versions':{'a':1},'hlm_item':{'id':'E0','name':'same'}})
    u2=replace(u1,logical_uid='unicode-2',metadata={**u1.metadata,'hlm_item':{'id':'E０','name':'same'}})
    publish(p,scope,a,x,y,r1,r2,u1,u2)
    assert MemoryViews(p).graph(o,w,s,aggregate=True)['display_total']==7


def test_canonical_validity_intervals_and_event_times_are_not_folded(env):
    p,c,o,w,s=env;scope=p.store.scope(o,w,'textual');a=raw('a','source',s)
    x=replace(profile('p1',a),valid_from_ms=100);y=replace(profile('p2',a),valid_from_ms=200)
    e1=replace(derived('e1','event',a,'episode'),business_timestamp_ms=100)
    e2=replace(derived('e2','event',a,'episode'),business_timestamp_ms=200)
    publish(p,scope,a,x,y,e1,e2)
    groups,_=group_records(MemoryViews(p).select(o,w,s,state_view='all')[0],o,w)
    assert len(groups)==5


def test_bidirectional_sources_use_same_real_edges_and_respect_eligibility(env):
    p,c,o,w,s=env;scope=p.store.scope(o,w,'textual');a=raw('raw','source',s);h=derived('h','higher',a)
    publish(p,scope,a,h,edges=(Edge('actual-edge','h','raw','SUPPORTED_BY'),))
    v=MemoryViews(p);up=v.relations(o,w,'raw');down=v.relations(o,w,'h')
    assert up['items'][0]['memory']['id']=='h' and up['items'][0]['direction']=='incoming'
    assert down['items'][0]['memory']['id']=='raw' and down['items'][0]['direction']=='outgoing'
    assert up['items'][0]['edge_id']==down['items'][0]['edge_id']
    publish(p,scope,replace(a,version=2,text_content='changed'))
    assert not v.relations(o,w,'raw')['items']
    assert v.relations(o,w,'raw',state_view='history')['items'][0]['memory']['id']=='h'


def test_deleted_record_is_auditable_without_reappearing_in_current_views(env):
    p,c,o,w,s=env;scope=p.store.scope(o,w,'textual');a=raw('raw','source',s);h=derived('h','knowledge',a);publish(p,scope,a,h)
    p.coordinator.apply(scope,'remove','remove',Mutation(delete_uids=('raw',)),{})
    assert not MemoryViews(p).graph(o,w,s)['nodes']
    assert c.get('/api/memories/raw',params={'workspace_id':w}).status_code==404
    audit=c.get('/api/memories/raw',params={'workspace_id':w,'audit':True});assert audit.status_code==200 and audit.json()['quality']['state']=='deleted'


def test_new_history_and_aggregation_endpoints_reject_cross_tenant(env):
    p,c,o,w,s=env;scope=p.store.scope(o,w,'textual');a=raw('private','CANARY-PROTECTED',s);publish(p,scope,a)
    login_test(c,p)
    for path in ('/api/memories/private/history','/api/memories/private/relations','/api/memories/private','/api/memory-groups/view-group:x','/api/memory-graph','/api/memory-schema'):
        r=c.get(path,params={'workspace_id':w,'session_id':s,'audit':True,'state_view':'all'})
        assert r.status_code==404 and 'CANARY-PROTECTED' not in r.text


def test_stored_revision_history_is_not_a_fabricated_old_snapshot(env):
    p,c,o,w,s=env
    payload={'workspace_id':w,'session_id':s,'request_id':'v34-stable-add','messages':[{'content':'我喜欢篮球。'}]}
    first=c.post('/api/memories',json=payload).json();key=first['memory_ids'][0]
    def ingest():
        job=p.store.claim('ingest','v34-test');assert job
        result=p.do_ingest(job);p.store.finish(job,result)
    ingest();assert c.post('/api/memories',json=payload).json()['memory_ids']==[key]
    assert p.store.claim('ingest','v34-test') is None
    r=c.put('/api/memories/'+key,params={'workspace_id':w},json={'request_id':'v34-stable-revision','expected_version':1,'content':'现在改为喜欢游泳。'});assert r.status_code==202;ingest()
    history=c.get('/api/memories/'+key+'/history',params={'workspace_id':w}).json()
    assert {v['version']:v['text'] for v in history['items']}=={1:'我喜欢篮球。',2:'现在改为喜欢游泳。'}
    assert history['items'][1]['kind']=='recorded_original_input'


def test_local_entity_labels_are_not_identity_keys_and_source_context_is_bounded(env):
    p,c,o,w,s=env;scope=p.store.scope(o,w,'textual');a=raw('a','两位同名人物',s)
    x=replace(derived('e0','相同显示名称',a,'entity'),metadata={'source_versions':{'a':1},'hlm_item':{'id':'E0','name':'同名','identity_context':'同句'}})
    y=replace(x,logical_uid='e1',metadata={**x.metadata,'hlm_item':{**x.metadata['hlm_item'],'id':'E1'}})
    publish(p,scope,a,x,y)
    g=MemoryViews(p).graph(o,w,s,memory_type='entity',relationship_view='provenance',limit=10,aggregate=True)
    assert {n['id'] for n in g['nodes']}=={'e0','e1','a'} and len(g['edges'])==2
    assert g['record_total']==2 and g['context_nodes_included']==1
    small=MemoryViews(p).graph(o,w,s,memory_type='entity',relationship_view='provenance',limit=2)
    assert small['page_size']==1 and small['truncated'] and len(small['nodes'])==2


def test_serialized_known_invalid_nodes_never_escape_current_graph(env,monkeypatch):
    p,c,o,w,s=env;scope=p.store.scope(o,w,'textual');a=raw('a','source',s);publish(p,scope,a)
    original=p.serialize
    def changed(*args,**kwargs):
        item=original(*args,**kwargs);item['eligible']=False;return item
    monkeypatch.setattr(p,'serialize',changed)
    assert MemoryViews(p).graph(o,w,s)['nodes']==[]


def test_alternative_source_refs_are_bidirectional_and_conversation_bound(env):
    p,c,o,w,s=env;scope=p.store.scope(o,w,'textual');a=raw('a','visual source',s)
    h=replace(derived('visual','actual visual representation',a,'visual_description'),metadata={'source_refs':[{'uid':'a','version':1}]})
    publish(p,scope,a,h)
    v=MemoryViews(p)
    assert {m.logical_uid for _,m,_ in v.select(o,w,s)[0]}=={'a','visual'}
    assert v.relations(o,w,'a')['items'][0]['memory']['id']=='visual'
    assert v.relations(o,w,'visual')['items'][0]['memory']['id']=='a'


def test_audit_can_follow_retired_real_edges_without_reenabling_them(env):
    p,c,o,w,s=env;scope=p.store.scope(o,w,'textual');a=raw('a','source',s)
    h=derived('h','historical higher',a,'episode');e=derived('e','another old representation',a,'entity')
    publish(p,scope,a,h,e,edges=(Edge('stored-semantic-edge','h','e','HAS_PARTICIPANT'),))
    p.coordinator.apply(scope,'retire-test','retire-test',Mutation(memories=(replace(h,active=False,version=2),replace(e,active=False,version=2)),delete_edge_ids=('stored-semantic-edge',)),{})
    v=MemoryViews(p)
    assert not v.relations(o,w,'h')['items']
    audit=v.relations(o,w,'h',state_view='all')
    assert any(x['memory']['id']=='e' and x['edge_id']=='stored-semantic-edge' for x in audit['items'])
