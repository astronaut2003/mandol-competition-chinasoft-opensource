"""Read-only quality views. Retrieval eligibility remains Platform.eligible.

No fact/UID is merged, no model is called, and no persistent index is changed.
"""
import json,re,unicodedata
from collections import Counter,defaultdict
from mandol_substrate.state import Scope
from mandol_substrate.state.models import digest,canonical_json
from .typed_memory import is_raw,view_type,MODES

STATES={
 'current':'当前有效', 'source_version_changed':'来源版本已变化',
 'source_missing':'来源已删除或不可用', 'profile_history':'画像历史状态',
 'deleted':'已撤回或删除', 'historical':'历史保留',
 'unpublished':'尚未完成发布', 'not_eligible':'不符合当前检索资格',
}


def logical_type(m,info):
    return view_type(m,MODES.get(m.metadata.get('memory_family'),info['mode']))


def validity(p,scope,m,eligible=None):
    """Authoritative boolean plus explanatory reasons, not a second gate."""
    good=p.eligible(scope,m) if eligible is None else eligible
    code='current' if good else 'not_eligible'
    if not good:
        if m.lifecycle=='deleted':code='deleted'
        elif not m.active:code='historical'
        elif m.commit_seq>p.state.scope_state(scope)['published_seq']:code='unpublished'
        else:
            profile=m.metadata.get('hlm_item',{}).get('profile_item',{})
            if profile.get('status') in ('historical','superseded','retracted','withdrawn'):code='profile_history'
            refs=m.metadata.get('source_versions',{}) or {r['uid']:r['version'] for r in m.metadata.get('source_refs',[])}
            sources={}
            for source_scope in (Scope('chinasoft.raw.v1',scope.user_id),scope):
                sources.update(p.state.get_memories(source_scope,list(refs)))
            if any(uid not in sources or not sources[uid].active for uid in refs):code='source_missing'
            elif any(sources[uid].version!=version for uid,version in refs.items()):code='source_version_changed'
    return {'eligible':bool(good),'state':code,'label':STATES[code],
            'authority':'Platform.eligible','index_readiness_is_separate':True}


def normalized(value):
    if isinstance(value,str):return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',value)).strip()
    if isinstance(value,dict):return {k:normalized(v) for k,v in value.items()}
    if isinstance(value,list):return [normalized(v) for v in value]
    return value


def semantic_key(scope,m,info):
    """Only qualified stable self profiles may fold across different sources.

    Other representations need identical provenance and structured content.
    Names, cosine similarity and text alone never establish entity identity.
    """
    if is_raw(m):return None
    item=m.metadata.get('hlm_item',{});profile=item.get('profile_item',{})
    validity_interval=(m.valid_from_ms,m.valid_to_ms)
    if m.memory_type=='profile' and profile.get('subject')=='user' and profile.get('status')=='current' and profile.get('lifetime')=='stable' and profile.get('attribute') and profile.get('value'):
        fields=('subject','kind','attribute','value','condition','scope','lifetime','status','time_qualifier','valid_from','valid_to','sensitive','personalization_purpose')
        return digest(['stable-self-profile-v34',scope.user_id,logical_type(m,info),validity_interval,{k:profile.get(k) for k in fields}])
    refs=m.metadata.get('source_versions',{})
    if not refs:return None
    structured=dict(item) if item else m.raw_data
    # Confidence is not an identity. Local entity IDs, identity_context,
    # relationship participants and temporal conditions are all retained.
    if item:structured={k:v for k,v in structured.items() if k!='confidence'}
    return digest(['same-evidence-v34',scope.key,logical_type(m,info),refs,
                   m.metadata.get('recipe_hash') or m.metadata.get('recipe_fingerprint'),
                   m.lifecycle,validity_interval,m.business_timestamp_ms,structured,m.text_content])


def group_records(records,owner,workspace):
    groups={};member_to_group={}
    for scope,m,info in records:
        key=semantic_key(scope,m,info)
        # Statuses/versions are never mixed in historical views either.
        state=(m.active,m.lifecycle,m.metadata.get('hlm_item',{}).get('profile_item',{}).get('status'),info.get('_quality',{}).get('state'))
        gid='view-group:'+digest([owner,workspace,key,state]) if key else m.logical_uid
        groups.setdefault(gid,[]).append((scope,m,info));member_to_group[m.logical_uid]=gid
    # A singleton always retains the real UID.
    result={}
    for gid,items in groups.items():
        key=gid if len(items)>1 else items[0][1].logical_uid
        result[key]=items
        for _,m,_ in items:member_to_group[m.logical_uid]=key
    return result,member_to_group


def labels(m,info):
    item=m.metadata.get('hlm_item',{});profile=item.get('profile_item',{})
    if profile:title=str(profile.get('statement') or profile.get('value') or m.text_content)
    elif m.memory_type=='entity':title=str(item.get('name') or m.text_content)
    elif m.memory_type=='relation':title=str(item.get('text') or m.text_content)
    elif m.memory_type=='code_symbol':title=str(m.raw_data.get('qualified_name') or m.raw_data.get('name') or m.text_content)
    else:title=m.text_content
    title=re.sub(r'^\[[^\]]+\]\s*','',title).strip().split('\n')[0][:180]
    # Width-aware CJK/Latin budget; at most two short lines, never body text.
    chars=[];width=0
    for ch in title:
        step=2 if unicodedata.east_asian_width(ch) in ('W','F') else 1
        if width+step>26:chars.append('…');break
        chars.append(ch);width+=step
    short=''.join(chars)
    return {'short':short,'title':title,'full_content_in_detail':True}


def diagnostics(p,records,owner,workspace):
    types=Counter();states=Counter();texts=defaultdict(list);signatures=defaultdict(list)
    for scope,m,info in records:
        types['basic' if is_raw(m) else 'higher']+=1
        states[validity(p,scope,m)['state']]+=1
        texts[(scope.user_id,logical_type(m,info),normalized(m.text_content))].append(m.logical_uid)
        if not is_raw(m):
            signatures[(scope.key,m.metadata.get('hlm_job_id') or m.request_id,logical_type(m,info),canonical_json(m.metadata.get('source_versions',{})),m.metadata.get('recipe_hash') or m.metadata.get('recipe_fingerprint'),normalized(m.text_content))].append(m.logical_uid)
    groups,_=group_records(records,owner,workspace)
    return {'counts':dict(types),'states':dict(states),'same_text_groups':sum(len(x)>1 for x in texts.values()),
            'same_text_records':sum(len(x) for x in texts.values() if len(x)>1),
            'same_job_source_recipe_text_candidate_groups':sum(len(x)>1 for x in signatures.values()),
            'conservative_groups':sum(len(x)>1 for x in groups.values()),
            'foldable_records':sum(len(x) for x in groups.values() if len(x)>1),
            'unique_uid_scope_pairs':len({(s.key,m.logical_uid) for s,m,_ in records}),
            'note':'Exact text is diagnostic only; groups are display containers, not entity resolution or stored fact merges.'}
