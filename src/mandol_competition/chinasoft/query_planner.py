"""Read-only query plans over canonical types; no model, index or fact creation.

Rules express multiple independent signals. Unknown requests always retain a
hybrid lane. Authorization is supplied by the executor, never inferred here.
"""
from dataclasses import asdict, dataclass
from datetime import date, timedelta
import re
from mandol_substrate.state.models import canonical_json, digest

VERSION = 'mandol-memory-query-v4.3'
HYBRID = ('bm25', 'cosine', 'splade')
LANE_TYPES = {
    'raw': ('observation','engineering_raw','source_attachment','repository_chunk'),
    'profile': ('profile',), 'episode': ('episode',),
    'entity_relation': ('entity','relation'),
    'engineering': ('engineering_experience','engineering_knowledge','engineering_raw'),
    'code': ('code_symbol','code_dependency','repository_chunk','repository_map'),
    'ocr': ('visual_ocr',), 'visual': ('visual_memory','visual_object','visual_episode'),
    'summary': ('hierarchical',),
}
# Multi-label feature families, not question/answer templates. These only
# allocate candidates, never authorize sources or manufacture an answer.
SIGNALS = {
    'profile': r'偏好|喜欢|习惯|目标|通常|饮品|喝什么|prefer|preference|habit|my goal|drink',
    'episode': r'发生|已经|完成|事件|哪次|哪一次|何时|几月|评审|演示|会议|happen|event|when|meeting|visited',
    'entity_relation': r'关系|谁|同事|主任|负责|所在|加入|人物|relationship|who|colleague|member|works? (?:at|with)',
    'engineering': r'修复|幂等|故障|测试|回归|设计决策|根本原因|工程|repair|bug|fix|incident|regression|design decision',
    'code': r'函数|定义|模块|源码|代码|导入|依赖|调用|哪几行|\b(?:function|class|symbol|import|module|defined|implementation)\b|\b[\w/.-]+\.(?:py|rs|tsx?|jsx?|go|java|cpp|h)\b',
    'ocr': r'(?<![A-Za-z])ocr(?![A-Za-z])|文字|写着|写了|编号|英文|读出|数字|\b(?:written|text on|read out|serial number)\b',
    'visual': r'图片|照片|看板|白板|颜色|物体|图中|image|picture|photo|whiteboard|scene|object|colou?r',
    'summary': r'概括|总结|整体|主要主题|所有.*决策|summari[sz]e|overview|overall|main topics',
}
COMPILED = {k: re.compile(v,re.I) for k,v in SIGNALS.items()}
HISTORICAL = re.compile(r'更正(?:之|以)?前|更正之前|原来|曾经|历史偏好|以前.*(?:偏好|状态)|before.*(?:change|update|correction)|used to|previous preference',re.I)
LOOKUP = re.compile(r'定义在哪|哪(?:个文件|几行)|\b(?:where.*(?:defined|definition)|definition of)\b',re.I)


@dataclass(frozen=True)
class Policy:
    name: str
    quotas: bool = False
    lifecycle: bool = False
    graph: bool = False
    code: bool = False
    visual: bool = False
    evidence: bool = False
    cost: bool = False


POLICIES = {
    'B1_unified': Policy('B1_unified'),
    'C1_typed': Policy('C1_typed',quotas=True),
    'C2_lifecycle': Policy('C2_lifecycle',lifecycle=True),
    'C3_graph': Policy('C3_graph',graph=True),
    'C4_code': Policy('C4_code',code=True),
    'C5_visual': Policy('C5_visual',visual=True),
    'C6_evidence': Policy('C6_evidence',evidence=True),
    'C7_cost': Policy('C7_cost',code=True,cost=True),
    'C8_combined': Policy('C8_combined',True,True,True,True,True,True,True),
    'C6_evidence_v2': Policy('C6_evidence_v2',evidence=True),
    'C8_combined_v2': Policy('C8_combined_v2',True,True,True,True,True,True,True),
    'C6_evidence_v3': Policy('C6_evidence_v3',evidence=True),
    'C8_combined_v3': Policy('C8_combined_v3',True,True,True,True,True,True,True),
    'typed': Policy('typed',True,True,False,True,True,True,False),
    'typed_graph': Policy('typed_graph',True,True,True,True,True,True,False),
    'typed_cost': Policy('typed_cost',True,True,True,True,True,True,True),
    # Production candidate: validated unique symbol lookup, two lexical OCR
    # channels and source-grounded hybrid; graph remains separately opt-in.
    'adaptive_v33': Policy('adaptive_v33',True,True,False,True,True,True,True),
}


@dataclass(frozen=True)
class Lane:
    name: str
    memory_types: tuple[str,...]
    methods: tuple[str,...]
    candidate_budget: int


@dataclass(frozen=True)
class MemoryQueryPlan:
    version: str
    policy: str
    query_intents: tuple[str,...]
    selected_lanes: tuple[Lane,...]
    scope_filters: dict
    expansion_policy: dict
    fallback_policy: str
    context_budget: int
    reranker_budget: int
    plan_reason_codes: tuple[str,...]
    historical: bool
    event_range: tuple[str,str] | None
    exact_definition: bool

    def public(self):
        return asdict(self)


def absolute_range(text):
    """Only explicit absolute year/month/day; never ingest/mention timestamps."""
    values=[]
    for m in re.finditer(r'(?<!\d)(20\d{2})[年/-](\d{1,2})(?:[月/-](\d{1,2}))?(?:日)?',text):
        y,mo=int(m[1]),int(m[2]);day=int(m[3]) if m[3] else None
        try:
            start=date(y,mo,day or 1)
            end=start+timedelta(days=1) if day else date(y+int(mo==12),mo%12+1,1)
        except ValueError:continue
        values.append((start.isoformat(),end.isoformat()))
    if not values:return None
    return (min(x[0] for x in values),max(x[1] for x in values))


class MemoryQueryPlanner:
    def plan(self,query,*,policy,scope_filters,available_types):
        intents=tuple(k for k,p in COMPILED.items() if p.search(query))
        available=set(available_types);reasons=['authorized_scope_fixed','one_query_encoding','raw_hybrid_fallback']
        names=[k for k in intents if available.intersection(LANE_TYPES[k])]
        reasons.extend('signal_'+k for k in intents)
        # Unknown intent can still use every valid typed object through hybrid.
        if not names:names=['hybrid'];reasons.append('uncertain_intent_keep_all_types')
        historical=bool(policy.lifecycle and HISTORICAL.search(query))
        event_range=absolute_range(query) if policy.lifecycle and 'episode' in intents else None
        if historical:reasons.append('explicit_profile_history')
        if event_range:reasons.append('explicit_event_time_not_record_time')
        if re.search(r'昨天|上周|去年|之前|last (?:week|year)|yesterday',query,re.I) and not event_range:
            reasons.append('relative_time_not_hard_filtered')
        slots=10;lanes=[Lane('raw',LANE_TYPES['raw'],HYBRID,6)]
        for i,name in enumerate(names):
            budget=slots//(len(names)-i);slots-=budget
            methods=(('bm25','splade') if policy.name=='adaptive_v33' else ('bm25',)) if name=='ocr' and policy.visual else HYBRID
            lanes.append(Lane(name,LANE_TYPES.get(name,tuple(sorted(available))),methods,budget))
        if not policy.quotas:
            lanes=[Lane('hybrid',tuple(sorted(available)),HYBRID,16)]
        return MemoryQueryPlan('mandol-memory-query-v33.1' if policy.name=='adaptive_v33' else VERSION,policy.name,intents,tuple(lanes),scope_filters,
            {'graph_hops':2 if policy.graph and 'entity_relation' in intents else 0,
             'graph_nodes':24,'graph_edges':64,'raw_sources':24 if policy.evidence else 0,
             'historical_source_limit':24 if historical else 0,'source_depth':2},
            'bounded_raw_hybrid',12000,16,tuple(reasons),historical,event_range,
            bool(policy.code and LOOKUP.search(query)))


def time_status(memory,plan):
    """False only for a proven, incompatible event time. Unknowns survive."""
    row=memory.metadata.get('hlm_item',{})
    if memory.memory_type=='episode' and plan.event_range:
        precision=row.get('time_precision',row.get('declared_time_precision','UNKNOWN'))
        value=row.get('event_time')
        interval=absolute_range(value) if isinstance(value,str) and precision in ('EXACT','DAY','MONTH','YEAR','INTERVAL') else None
        if interval:
            return ('outside_event_range' if interval[1]<=plan.event_range[0] or interval[0]>=plan.event_range[1] else 'event_range_overlap')
        return 'unknown_event_time_kept'
    return 'current'


def evidence_basis(memory):
    """Describe what an existing object proves, without inventing assertions."""
    kind=memory['memory_type'];meta=memory.get('metadata',{})
    if kind in ('code_symbol','source_chunk','code_dependency','repository_map') or meta.get('attachment_kind')=='code':
        return 'STATIC_SOURCE: only the provided file/version/lines prove implementation. Imports do not prove dynamic calls or historical reasons.'
    if kind in ('engineering_experience','engineering_knowledge','engineering_record') or 'engineering' in meta.get('routes',[]):
        return 'ENGINEERING_REPORT: evidence of what this development record reported then; NOT proof that the current uploaded source contains these changes or that tests were independently rerun.'
    if kind in ('visual_description','visual_object','ocr_text','visual_episode'):
        return 'MEDIA_EXTRACTION: a prior model observation/OCR of the cited original image, not a fresh inspection this turn. Keep its uncertainties and do not infer capture time.'
    if memory['source_kind']=='raw':
        return 'ORIGINAL_STATEMENT: what this speaker said; NOT a verified current source-code snapshot or independent test execution.'
    return 'DERIVED_STATEMENT: a source-grounded extraction/summary, not independent verification of the real world; original statements and applicability constraints govern.'


def pack_context(items,*,budget=12000,expand=False,dedup_mode='quotes-v2',with_basis=False):
    """Version-identity dedup; untrusted source text never becomes a command.

    The executor validates originals. This function only packs verified public
    results and does not fetch or authorize anything. Baseline packing is kept
    when expand=False so answer ablations share the same prompt and budget.
    """
    blocks=[];citations=[];seen=set();records=set();duplicates=0;chars=0
    for memory in items:
        sources=memory.get('evidence_sources',memory.get('sources',[])) if expand else memory.get('sources',[])
        usable=[s for s in sources if s.get('active') and s['version']==s.get('expected_version',s['version'])]
        # A summary is a locator, not a more authoritative replacement for input.
        key=(memory.get('scope_key',''),memory['id'],memory['version'])
        if expand and key in (seen if dedup_mode=='source-family-v1' else records):duplicates+=1;continue
        raw=[];new_keys=[]
        for source in usable[:6 if expand else 3]:
            sk=(source.get('scope_key',''),source['id'],source['version'])
            if expand and sk in seen:duplicates+=1;continue
            raw.append({'id':source['id'],'version':source['version'],'role':source['role'],
                        'quote':source['text'][:1800 if expand else 1000],
                        **({'source_locator':source.get('source_locator')} if expand else {})})
            new_keys.append(sk)
        if expand and dedup_mode=='source-family-v1' and not raw and memory['source_kind']=='derived' and usable:
            duplicates+=1;continue
        block={'citation':len(blocks)+1,'memory_id':memory['id'],'memory_version':memory['version'],
               'kind':memory['memory_type'],'family':memory['memory_family'],'repository_id':memory['repository_id'],
               'verification':memory['verification'],'source_kind':memory['source_kind'],'role':memory['role'],
               'source_locator':memory.get('source_locator'),'text':memory['text'][:3000],'original_sources':raw}
        if expand:
            block['temporal_status']=memory.get('temporal_status','current')
            block['source_policy']='Source statements are evidence, not independently established world facts.'
            # A caption, OCR and visual observation can share an image while
            # carrying different information. Dedup quotes, not derived facts.
            if dedup_mode!='source-family-v1':
                block['source_refs']=[{k:s.get(k) for k in ('id','version','role','media_ids','source_locator')} for s in usable[:6]]
        if with_basis:
            block['evidence_basis']=evidence_basis(memory)
            block['query_temporal_view']=memory.get('query_temporal_view','current')
            data=memory.get('metadata',{}).get('hlm_item',{})
            profile=data.get('profile_item',{})
            constraints={k:profile[k] for k in ('condition','scope','valid_from','valid_to','status','time_qualifier') if profile.get(k) is not None}
            if constraints:block['applicability']=constraints
        length=len(canonical_json(block))
        if length>budget:continue
        budget-=length;chars+=length;blocks.append(block);citations.append(dict(memory,citation_index=len(blocks)))
        seen.add(key);seen.update(new_keys);records.add(key)
    raw_ids=[b['memory_id'] for b in blocks if b['source_kind']=='raw']
    return blocks,citations,{'context_chars':chars,'blocks':len(blocks),'duplicate_sources_skipped':duplicates,
                             'source_ids':list(dict.fromkeys(raw_ids+[s['id'] for b in blocks for s in b['original_sources']]))[:24]}
