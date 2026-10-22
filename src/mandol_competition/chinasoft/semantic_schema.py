"""Read-only semantic schema over the existing canonical type registry.

Support is a capability, not evidence that this conversation instantiated it.
Counts are scoped, source-version checked and explicitly bounded. No database
engine or semantic relationship is created by this explanatory view.
"""
from collections import Counter,defaultdict
import time,copy
from .typed_memory import TYPE_MAP,view_type,is_raw
from .visualization import MemoryViews

VERSION='mandol-semantic-schema-v34.1'
SPECS={
 'conversation_message':('对话消息','用户或助手实际输入的基础记录。','message',('role','content','session_id'),('recent_messages','bm25','dense','splade'),'raw_ingestion'),
 'source_attachment':('代码与文档文件','已校验、版本化的上传文件文本。','attachment',('name','hash','parser','version','line_range'),('file_lines','bm25','dense','splade'),'attachment-ingestion-v31'),
 'engineering_record':('工程过程记录','开发问题、处理过程与验证陈述的基础记录。','message',('role','session_id','routes'),('bm25','dense','splade'),'raw_ingestion'),
 'image_text':('图片及说明','原图引用和用户说明；说明不证明图像事实。','image',('media_ids','role','content_hash'),('media_access','bm25','dense','splade'),'raw_ingestion'),
 'source_chunk':('仓库源码片段','授权仓库不可变版本的有行号源码。','code',('path','git_head','hash','start_line','end_line'),('path_lookup','bm25','dense','splade'),'repository-parser'),
 'user_profile':('用户画像与偏好','仅从适格用户陈述提取，保留条件、更正与生命周期。','conversation',('slot','value','condition','valid_from','valid_to','status'),('lifecycle_filter','bm25','dense','splade'),'conversation-v3'),
 'episodic_event':('情景事件','来源支持的事件、计划和状态；未知事件时间不补写。','conversation',('event_time','time_precision','participants','event_status'),('event_time_filter','bm25','dense','splade'),'conversation-v3'),
 'entity':('实体','上下文中的实体提及，同名不自动合并。','conversation',('name','entity_type','source_versions'),('entity_candidates','stored_edges','bm25','dense','splade'),'conversation-v3'),
 'relation':('实体关系','带原文证据的关系节点和参与实体。','conversation',('subject','predicate','object','evidence'),('stored_edges','bm25','dense','splade'),'conversation-v3'),
 'conversation_summary':('会话摘要','定位主题的来源约束摘要，不取代基础证据。','conversation',('source_versions','summary','session_id'),('bm25','dense','splade','source_expansion'),'conversation-v3'),
 'engineering_experience':('工程经验','由足量适格工程记录提炼的问题、方案和验证报告。','engineering',('problem','solution','validation','source_refs'),('bm25','dense','splade','source_expansion'),'engineering-experience'),
 'engineering_knowledge':('工程知识','带来源的工程决策和约束，不从代码臆测历史动机。','engineering',('source_refs','recipe_fingerprint','code_associations'),('bm25','dense','splade','source_expansion'),'engineering-experience'),
 'code_symbol':('代码符号','确定性解析的类、函数和方法。','code',('name','qualified_name','path','start_line','end_line'),('symbol_lookup','path_lookup','bm25','dense','splade'),'repository/attachment-parser'),
 'code_dependency':('导入与依赖','可静态识别的导入结构，不代表动态调用。','code',('path','target_path','resolved','git_head'),('path_lookup','stored_edges','bm25','dense','splade'),'repository-parser'),
 'repository_map':('仓库结构摘要','由实际文件和符号生成的有界 Repository Map。','code',('repository_id','generation','file_count','symbol_count'),('path_lookup','bm25','dense','splade'),'repository-parser'),
 'visual_description':('视觉描述','真实 VLM 对原图的可见内容描述，保留不确定性。','visual',('description','media_ids','uncertainties'),('bm25','dense','splade','source_expansion'),'visual-scenario-v2'),
 'ocr_text':('OCR 文字','模型在图片中识别的文字，不构成用户个人陈述。','visual',('items','media_ids','uncertainties'),('bm25','splade','source_expansion'),'visual-scenario-v2'),
 'visual_object':('可见对象','图片对象和属性的模型观察，不自动确认对象身份。','visual',('items','media_ids','uncertainties'),('bm25','dense','splade'),'visual-scenario-v2'),
 'visual_episode':('视觉会话窗口','按真实来源组织的视觉窗口，不推断照片间因果。','visual',('source_versions','session_id','media_ids'),('bm25','dense','splade','source_expansion'),'visual-navigation'),
}
BASE={'conversation_message','source_attachment','engineering_record','image_text','source_chunk'}


def overview(platform,owner,workspace,session=None,repository=None,state_view='all'):
    platform.store.workspace(owner,workspace)
    views=MemoryViews(platform)
    scoped=views.scopes(owner,workspace,repository)
    if session:platform.store.session(owner,workspace,session)
    args=[owner,workspace];where='j.owner=? AND j.workspace=?'
    if session:
        where+=' AND j.ingestion IN (SELECT id FROM cs_ingestions WHERE owner=? AND workspace=? AND session=?)';args.extend([owner,workspace,session])
    jobs=platform.store.rows('SELECT kind,status,COUNT(*) count,MAX(updated_at) last_update FROM cs_jobs j WHERE '+where+' GROUP BY kind,status',args)
    keys=[s.key for s,_ in scoped]
    frontier_token=platform.state._rows('SELECT s.scope,s.published_seq,p.published_seq vector_seq,p.status FROM scopes s LEFT JOIN projection_frontiers p ON p.scope=s.scope AND p.channel=? WHERE s.scope IN ('+','.join('?' for _ in keys)+') ORDER BY s.scope',[platform.projection_channel,*keys]) if keys else []
    lifecycle_token=[platform.store.rows('SELECT id,active FROM cs_attachments WHERE owner=? AND workspace=?',(owner,workspace)),platform.store.rows('SELECT id,deleted FROM cs_sessions WHERE owner=? AND workspace=?',(owner,workspace)),platform.store.rows('SELECT id,deleted,generation FROM cs_repositories WHERE owner=? AND workspace=?',(owner,workspace))]
    cache_key=(owner,workspace,session,repository,state_view,repr(frontier_token),repr(jobs),repr(lifecycle_token))
    cache=getattr(platform,'schema_overview_cache',None)
    if cache is None:cache={};platform.schema_overview_cache=cache
    cached=cache.get(cache_key)
    if cached and time.monotonic()-cached[0]<5:
        result=copy.deepcopy(cached[1]);result['cache_age_ms']=round((time.monotonic()-cached[0])*1000);return result
    records,total=views.select(owner,workspace,session,repository,limit=5000,state_view='all')
    all_records=records
    if state_view not in ('current','history','all'):
        from .store import AppError
        raise AppError('无效记忆状态视图')
    records=[r for r in records if state_view=='all' or r[2]['_quality']['eligible']==(state_view=='current')]
    rows={};by_type=defaultdict(list);states=Counter();scope_states={}
    for scope,m,info in records:
        mode=m.metadata.get('memory_family')
        mode={'conversation':'textual','code':'code','multimodal':'multimodal'}.get(mode,info['mode'])
        kind=view_type(m,mode);valid=platform.eligible(scope,m)
        if scope.key not in scope_states:scope_states[scope.key]=platform.state.scope_state(scope)
        state=scope_states[scope.key]
        if scope.key not in rows:
            frontier=platform.state._rows('SELECT published_seq,status FROM projection_frontiers WHERE scope=? AND channel=?',(scope.key,platform.projection_channel))
            rows[scope.key]=bool(frontier and frontier[0]['status']=='ready' and frontier[0]['published_seq']==state['published_seq'])
        by_type[kind].append((m,valid,rows[scope.key]));states['basic' if is_raw(m) else 'higher']+=1
    supported={v for mapping in TYPE_MAP.values() for v in mapping.values()}
    types=[]
    for kind in sorted(supported|set(by_type)):
        name,definition,group,attrs,methods,recipe=SPECS.get(kind,(kind,'现有记录类型；暂无专属能力描述。','other',('source_versions',),('bm25','dense','splade'),'stored-record'))
        actual=by_type[kind];valid=sum(v for _,v,_ in actual);ready=sum(v and r for _,v,r in actual)
        recipes=sorted({m.metadata.get('recipe_fingerprint') or m.metadata.get('recipe_hash') for m,_,_ in actual if m.metadata.get('recipe_fingerprint') or m.metadata.get('recipe_hash')})[:3]
        physical_types=sorted({physical for mapping in TYPE_MAP.values() for physical,logical in mapping.items() if logical==kind})
        source_types={'user_profile':['conversation_message'],'episodic_event':['conversation_message'],'entity':['conversation_message'],'relation':['conversation_message'],
            'conversation_summary':['conversation_message'],'engineering_experience':['engineering_record','conversation_message'],
            'engineering_knowledge':['engineering_record','conversation_message'],'code_symbol':['source_attachment','source_chunk'],
            'code_dependency':['source_chunk'],'repository_map':['source_chunk','code_symbol'],
            'visual_description':['image_text'],'ocr_text':['image_text'],'visual_object':['image_text'],'visual_episode':['image_text']}.get(kind,[])
        types.append({'id':kind,'name':name,'definition':definition,'group':group,'source_kind':'raw' if kind in BASE else 'derived',
            'attributes':attrs,'methods':methods,'recipe':recipe,'observed_recipe_fingerprints':recipes,'physical_types':physical_types,'source_types':source_types,
            'source_constraints':'同账号/工作区，来源 UID、版本及生命周期有效；生成结论仍需语义审核。',
            'storage':['SQLite memories + versions','SQLite source_versions / graph_edges','SQLite BM25','Qdrant dense / SPLADE'],
            'supported':kind in supported,'count':len(actual),'valid':valid,'invalid':len(actual)-valid,'indexed':ready,
            'sample_ids':[m.logical_uid for m,_,_ in actual[:3]],'state':'not_instantiated' if not actual else 'indexing' if ready<valid else 'ready' if valid else 'invalid'})
    health=getattr(platform.retrieval,'health_state',{})
    models={name:{k:v for k,v in data.items() if k in ('ready','reason','checked_at')} for name,data in health.items()}
    result={'version':VERSION,'workspace_id':workspace,'session_id':session,'scope':'conversation' if session else 'conversations','types':types,
        'instance':{'total_candidates':total,'counted':len(records),'complete':views.selection_info['complete'],'basic':states['basic'],'higher':states['higher'],
                    'valid':sum(t['valid'] for t in types),'invalid':sum(t['invalid'] for t in types)},
        'jobs':jobs,'retrieval':{'profile':platform.vector_profile.hash,'strategy':platform.settings.memory_query_strategy,
            'published_scopes':len(rows),'vector_ready_scopes':sum(rows.values()),'models_last_probe':models,
            'methods':['Dense','SPLADE','BM25','RRF','BGE'],'model_probe_note':'最近服务探测；投影状态与推理服务状态分别记录。'},
        'storage':{'canonical':'SQLite 是持久化真值，版本与来源保存在 canonical records。','lexical':'SQLite BM25 倒排索引。',
                   'vector':'Qdrant 是同一 Profile 的 Dense/SPLADE 投影，发布 frontier 校验可检索状态。',
                   'graph':'SQLite graph_edges 保存显式关系；分组仅用于显示，不合并 UID。'},
        'relation_definitions':[{'name':'SUPPORTED_BY / DERIVED_FROM','category':'provenance','meaning':'来源依赖，不表示现实实体关系。'},
            {'name':'RELATION_SUBJECT / RELATION_OBJECT','category':'semantic','meaning':'已存关系节点的参与实体，需结合关系正文和证据。'},
            {'name':'STATIC_IMPORT / CONTAINS','category':'structure','meaning':'确定性代码/容器结构，不能据此推断动态调用或历史原因。'}],
        'state_view':state_view,'scope_totals':{'states':dict(Counter(i['_quality']['state'] for _,_,i in all_records)),'records':len(all_records),'current':sum(i['_quality']['eligible'] for _,_,i in all_records),'history':sum(not i['_quality']['eligible'] for _,_,i in all_records)},
        'computed_at':int(time.time()*1000),'cache_age_ms':0,'cache_ttl_ms':5000}
    while len(cache)>=64:cache.pop(next(iter(cache)),None)
    cache[cache_key]=(time.monotonic(),copy.deepcopy(result))
    return result
