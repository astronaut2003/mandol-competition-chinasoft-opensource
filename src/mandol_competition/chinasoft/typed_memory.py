"""Scenario views over existing canonical types, without a storage rewrite."""
from dataclasses import dataclass, asdict
from .store import AppError

FAMILIES = {'textual':'conversation', 'code':'code', 'multimodal':'multimodal'}
MODES = {v:k for k,v in FAMILIES.items()}
TYPE_MAP = {
    'textual': {'source_attachment':'source_attachment','observation':'conversation_message','hierarchical':'conversation_summary',
                'episode':'episodic_event','entity':'entity','relation':'relation','profile':'user_profile'},
    'code': {'engineering_raw':'engineering_record','engineering_experience':'engineering_experience',
             'engineering_knowledge':'engineering_knowledge','repository_chunk':'source_chunk',
             'code_symbol':'code_symbol','code_dependency':'code_dependency','repository_map':'repository_map'},
    'multimodal': {'observation':'image_text','visual_memory':'visual_description',
                   'visual_episode':'visual_episode','visual_ocr':'ocr_text','visual_object':'visual_object'},
}

def view_type(memory, mode):
    # Unified conversations keep one canonical raw Scope. Attachments retain
    # their semantic input shape in the application view, without rewriting it.
    if memory.memory_type=='observation' and memory.metadata.get('media_ids'):return 'image_text'
    return TYPE_MAP.get(mode,{}).get(memory.memory_type, memory.memory_type)

def is_raw(memory):
    return memory.memory_level == 'L0_Observation' and memory.role != 'derived'

@dataclass(frozen=True)
class QueryPlan:
    workspace_id: str
    family: str
    repository_id: str
    retrieval_scope: str
    preferred_types: tuple[str, ...]

    def public(self): return asdict(self)

def plan_query(workspace, mode, repository_id, retrieval_scope, query):
    mode = MODES.get(mode, mode or 'textual')
    if mode not in FAMILIES: raise AppError('无效记忆场景')
    if retrieval_scope not in ('current','workspace'): raise AppError('必须明确选择 current 或 workspace 检索范围')
    if repository_id and mode != 'code': raise AppError('Repository 仅属于 Code 场景')
    q=query.lower()
    if mode=='textual':
        if any(x in q for x in ('偏好','喜欢','目标','preference','profile')): preferred=('user_profile','conversation_message')
        elif any(x in q for x in ('关系','人物','谁','relationship')): preferred=('relation','entity','conversation_message')
        elif any(x in q for x in ('事件','发生','过去','历史','event')): preferred=('episodic_event','conversation_message')
        else: preferred=('conversation_message','conversation_summary','user_profile')
    elif mode=='code':
        if any(x in q for x in ('此前','修复','历史','经验','设计决策')): preferred=('engineering_experience','engineering_knowledge','engineering_record')
        elif any(x in q for x in ('依赖','引用','调用','import','reference','depend')): preferred=('code_dependency','code_symbol','source_chunk')
        elif any(x in q for x in ('主要模块','仓库结构','repository map')): preferred=('repository_map','code_symbol')
        else: preferred=('code_symbol','source_chunk','engineering_experience')
    else:
        if any(x in q for x in ('文字','ocr','读出','写着')): preferred=('ocr_text','visual_description','image_text')
        elif any(x in q for x in ('物体','对象','颜色','object')): preferred=('visual_object','visual_description','image_text')
        else: preferred=('visual_description','visual_episode','image_text')
    return QueryPlan(workspace,FAMILIES[mode],repository_id or '',retrieval_scope,preferred)
