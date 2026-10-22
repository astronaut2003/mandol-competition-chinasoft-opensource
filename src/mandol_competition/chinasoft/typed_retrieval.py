"""Optional query execution over existing SQL/Qdrant publication contracts.

No schema changes or extra persistent index. All traversal and source expansion
is confined to server-authorized scopes and revalidated after remote reranking.
"""
from collections import defaultdict
from dataclasses import replace
import json
import re
import time

from mandol_substrate.state import Scope
from mandol_substrate.state.models import digest
from .models import ModelError
from .store import AppError
from .typed_memory import FAMILIES, MODES, TYPE_MAP, plan_query
from .query_planner import HYBRID, POLICIES, MemoryQueryPlanner, time_status

MAX_SCOPE_CATALOG = 5000
MAX_TOTAL_CATALOG = 10000

def qualified_code_hint(platform,owner,workspace,query,session_id=None,repository_id=None):
    """Only an actual authorized definition can avoid the semantic router call."""
    from .query_planner import LOOKUP
    if not LOOKUP.search(query):return None
    terms=sorted({t for t in re.findall(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*',query) if len(t)>=3 and t.lower() not in {'where','which','defined','definition','function','class','file','method'}})[:12]
    if not terms:return None
    from .visualization import MemoryViews
    views=MemoryViews(platform);found=[]
    for scope,info in platform.store.scopes(owner,workspace):
        if info['repository'] and info['repository']!=repository_id:continue
        marks=','.join('?' for _ in terms);seq=platform.state.scope_state(scope)['published_seq']
        rows=platform.state._rows("SELECT logical_uid FROM memories WHERE scope=? AND active=1 AND commit_seq<=? AND memory_type='code_symbol' AND (json_extract(raw_data,'$.name') IN ("+marks+") OR json_extract(raw_data,'$.qualified_name') IN ("+marks+")) LIMIT 8",[scope.key,seq,*terms,*terms])
        for m in platform.state.get_memories(scope,[r['logical_uid'] for r in rows]).values():
            if not info['repository'] and session_id and not views.belongs(scope,m,session_id):continue
            if platform.eligible(scope,m):found.append(m.logical_uid)
    return {'matches':len(found),'basis':'authorized_current_code_symbols'} if found else None


class TypedRetrieval:
    def __init__(self,platform):
        self.p=platform
        self.objects={};self.validity={};self.descriptors={};self.scopes={};self.source_descriptors={}
        self.rejected=set();self.source_reads=0;self.sql_queries=0
        self.expanded_sources={}

    def load(self,scope,ids):
        missing=list(dict.fromkeys(u for u in ids if (scope.key,u) not in self.objects))
        if missing:
            values=self.p.state.get_memories(scope,missing);self.sql_queries+=1
            for uid in missing:self.objects[(scope.key,uid)]=values.get(uid)
        return {uid:self.objects[(scope.key,uid)] for uid in ids if self.objects.get((scope.key,uid))}

    def allowed_object(self,scope,memory):
        if scope.key not in self.scopes:return False
        if (scope.key,memory.logical_uid) not in self.source_descriptors:return False
        return True

    def valid(self,scope,memory,*,fresh=False):
        key=(scope.key,memory.logical_uid,memory.version)
        if not self.allowed_object(scope,memory):return False
        if not fresh and key in self.validity:return self.validity[key]
        valid=self.p.eligible(scope,memory)
        if not valid and self.plan.historical and memory.memory_type=='profile':
            # Historical Profile is a navigation object only. Final citations
            # become its actual still-valid raw records, never an ineligible fact.
            if memory.active and self.p.visible(scope,memory) and memory.commit_seq<=self.p.state.scope_state(scope)['published_seq']:
                p=self.p
                reader=p.v3_hlm if memory.metadata.get('recipe_hash')==p.v3_recipe.fingerprint else p.hlm if memory.metadata.get('recipe_hash')==p.recipe.fingerprint else p.legacy_hlm
                valid=memory.logical_uid in reader.eligible_history(Scope('chinasoft.raw.v1',scope.user_id),{memory.logical_uid:memory})
        if valid and self.policy.lifecycle and time_status(memory,self.plan)=='outside_event_range':valid=False
        if not valid:self.rejected.add((scope.key,memory.logical_uid))
        self.validity[key]=valid
        return valid

    def resolve_sources(self,scope,memory,depth=0,visited=None):
        if depth>2:return []
        versions=memory.metadata.get('source_versions') or {r['uid']:r['version'] for r in memory.metadata.get('source_refs',[])}
        if len(versions)>64:return []
        visited=set(visited or ())|{(scope.key,memory.logical_uid)};result=[]
        # Sources are same-user, either canonical raw or the current repository
        # partition. Merely finding a UID in some other workspace is forbidden.
        raw=Scope('chinasoft.raw.v1',scope.user_id)
        source_scopes=list({s.key:s for s in (scope,raw) if s.key in self.scopes}.values())
        for uid,version in versions.items():
            source_limit=max(self.plan.expansion_policy['raw_sources'],self.plan.expansion_policy.get('historical_source_limit',0))
            if len(self.expanded_sources)>=source_limit:
                # The budget is query-wide, not multiplied by every hit. Known
                # originals can be shared without adding a new source payload.
                known=[(s,m) for (sk,u,v),(s,m) in self.expanded_sources.items() if u==uid and v==version and sk in {ss.key for ss in source_scopes}]
                if len(known)==1:result.extend(known)
                continue
            found=[]
            for source_scope in source_scopes:
                source=self.load(source_scope,[uid]).get(uid);self.source_reads+=1
                if source and source.version==version and (source_scope.key,uid) not in visited and self.allowed_object(source_scope,source) and self.p.eligible(source_scope,source):
                    found.append((source_scope,source))
            if len(found)!=1:continue
            source_scope,source=found[0]
            if source.memory_level=='L0_Observation' and source.role!='derived':
                result.append((source_scope,source))
                self.expanded_sources[(source_scope.key,source.logical_uid,source.version)]=(source_scope,source)
            else:result.extend(self.resolve_sources(source_scope,source,depth+1,visited))
            if len(result)>=24:break
        return list({(s.key,m.logical_uid,m.version):(s,m) for s,m in result}.values())[:24]

    def exact(self,query):
        result=[]
        terms=set(re.findall(r'[A-Za-z_][A-Za-z_0-9]*',query))
        paths=set(re.findall(r'[\w./-]+\.(?:py|rs|tsx?|jsx?|go|java|cpp|h)\b',query))
        for key,row in self.descriptors.items():
            if row['memory_type']!='code_symbol':continue
            locator=row['metadata'].get('source_locator') or {}
            if row.get('symbol_name') in terms or row.get('qualified_name') in terms or locator.get('path') in paths:
                result.append(key)
        for scope,info in self.scopes.values():
            if info['repository']:
                result.extend((scope.key,u) for u in self.p.repositories.exact_candidates(info['repository'],query) if (scope.key,u) in self.descriptors)
        return list(dict.fromkeys(result))[:24]

    def graph(self,seeds):
        """Bounded traversal of stored structural edges, never name-based joins."""
        relations=('SUBJECT','OBJECT','RELATION_SUBJECT','RELATION_OBJECT','HAS_PARTICIPANT','PARTICIPATES_IN','IMPORTS_SYMBOL','RELATED_CODE')
        front=list(seeds)[:4];seen=set(front);found=[];edges_seen=0;paths=[]
        def resolve(sk,uid):
            scope=self.scopes[sk][0];m=self.load(scope,[uid]).get(uid)
            if not m or m.lifecycle=='deleted' or m.commit_seq>self.p.state.scope_state(scope)['published_seq'] or not self.p.visible(scope,m):return None
            ref=m.metadata.get('hlm_reference')
            if ref:
                target=self.scopes.get(ref.get('scope_key'))
                if not target or target[0].user_id!=scope.user_id:return None
                from .visualization import MemoryViews
                if MemoryViews(self.p).resolve_reference(scope,uid,self.scopes)!=ref['uid']:return None
                t=self.load(target[0],[ref['uid']]).get(ref['uid'])
                if not t or t.version!=ref['version'] or digest(t.text_content)!=ref['text_hash']:return None
                return (target[0].key,t.logical_uid) if self.valid(target[0],t) else None
            return (sk,uid) if m.active and self.valid(scope,m) else None
        for depth in range(self.plan.expansion_policy['graph_hops']):
            next_front=[]
            for sk,uid in front:
                aliases=[(sk,uid)]
                # Reference aliases are stored navigation objects, not semantic
                # identity guesses. Only exact scope/UID/version/hash resolves.
                for other_scope,_ in self.scopes.values():
                    if other_scope.user_id!=self.scopes[sk][0].user_id:continue
                    refs=self.p.state._rows("SELECT logical_uid FROM memories WHERE scope=? AND (active=1 OR (memory_type='canonical_reference' AND memory_level='Reference' AND lifecycle<>'deleted')) AND json_extract(metadata,'$.hlm_reference.scope_key')=? AND json_extract(metadata,'$.hlm_reference.uid')=? LIMIT 8",(other_scope.key,sk,uid));self.sql_queries+=1
                    aliases.extend((other_scope.key,r['logical_uid']) for r in refs if resolve(other_scope.key,r['logical_uid'])==(sk,uid))
                for ask,alias in aliases:
                    remaining=64-edges_seen
                    if remaining<=0:break
                    # OR used only the scope prefix in the measured SQLite
                    # plan. UNION keeps both existing endpoint indexes selective.
                    seq=self.p.state.scope_state(self.scopes[ask][0])['published_seq']
                    base='SELECT source_uid,target_uid,relation_type FROM graph_edges WHERE scope=? AND active=1 AND commit_seq<=? AND relation_type IN ('+','.join('?' for _ in relations)+')'
                    rows=self.p.state._rows(base+' AND source_uid=? UNION ALL '+base+' AND target_uid=? AND source_uid<>? LIMIT ?',
                        [ask,seq,*relations,alias,ask,seq,*relations,alias,alias,remaining]);self.sql_queries+=1
                    for edge in rows:
                        edges_seen+=1
                        a=resolve(ask,edge['source_uid']);b=resolve(ask,edge['target_uid'])
                        if not a or not b:continue
                        for target in (a,b):
                            if target not in seen:
                                seen.add(target);next_front.append(target);found.append(target)
                                if len(paths)<8:paths.append({'from':uid,'to':target[1],'relation':edge['relation_type'],'depth':depth+1})
                                if len(found)>=24:return found,edges_seen,paths
            front=next_front
            if not front or edges_seen>=64:break
        return found,edges_seen,paths

    def search(self,owner,workspace,query,mode=None,limit=8,*,strategy,repository_id=None,retrieval_scope='current',memory_type=None,source_kind=None,session_id=None,conversation_first=False,exclude_sources=None):
        start=time.monotonic();timings={};p=self.p
        self.policy=POLICIES[strategy]
        if source_kind not in (None,'','raw','derived'):raise AppError('无效来源分类')
        if not 1<=limit<=12:raise AppError('检索数量必须在 1–12 以内')
        mode=MODES.get(mode,mode or 'textual');legacy=plan_query(workspace,mode,repository_id,retrieval_scope,query)
        if repository_id:p.store.repository(owner,workspace,repository_id)
        if session_id:p.store.session(owner,workspace,session_id)
        scopes=p.store.scopes(owner,workspace,None if retrieval_scope=='workspace' or conversation_first else mode,
            None if retrieval_scope=='workspace' or conversation_first else repository_id or '')
        if conversation_first:scopes=[(s,i) for s,i in scopes if not i['repository'] or i['repository']==repository_id]
        self.scopes={s.key:(s,i) for s,i in scopes};excluded=set(exclude_sources or ())
        # Bounded metadata reads, no whole source text / media transferred here.
        catalog_rows=0
        for scope,info in scopes:
            seq=p.state.scope_state(scope)['published_seq']
            read_limit=min(MAX_SCOPE_CATALOG,MAX_TOTAL_CATALOG-catalog_rows)+1
            rows=p.state._rows("SELECT logical_uid,memory_type,memory_level,role,session_id,metadata,json_extract(raw_data,'$.name') symbol_name,json_extract(raw_data,'$.qualified_name') qualified_name FROM memories WHERE scope=? AND active=1 AND commit_seq<=? AND memory_type NOT IN ('reference','canonical_reference') LIMIT ?",(scope.key,seq,read_limit));self.sql_queries+=1
            catalog_rows+=len(rows)
            if len(rows)>MAX_SCOPE_CATALOG or catalog_rows>MAX_TOTAL_CATALOG:
                result=p._search_baseline(owner,workspace,query,mode,limit,repository_id=repository_id,retrieval_scope=retrieval_scope,memory_type=memory_type,source_kind=source_kind,session_id=session_id,conversation_first=conversation_first,exclude_sources=exclude_sources)
                result['route']['planner_fallback']='typed_metadata_budget_exceeded';return result
            deleted={r['id'] for r in p.store.rows('SELECT id FROM cs_sessions WHERE owner=? AND workspace=? AND deleted=1',(owner,workspace))}
            for row in rows:
                row['metadata']=json.loads(row['metadata']);refs=row['metadata'].get('source_versions') or {r['uid']:r['version'] for r in row['metadata'].get('source_refs',[])}
                if row['logical_uid'] in excluded or excluded.intersection(refs) or row['session_id'] in deleted:continue
                self.source_descriptors[(scope.key,row['logical_uid'])]=row
                if memory_type and memory_type not in (row['memory_type'],TYPE_MAP.get(MODES.get(row['metadata'].get('memory_family'),info['mode']),{}).get(row['memory_type'])):continue
                raw=row['memory_level']=='L0_Observation' and row['role']!='derived'
                if source_kind and raw!=(source_kind=='raw'):continue
                self.descriptors[(scope.key,row['logical_uid'])]=row
        # Session membership checks include *all* originals, not just one SQL
        # EXISTS witness. Provenance expansion may read raw despite a derived-only
        # result filter, but it never leaves the authorized conversation.
        from .visualization import MemoryViews
        views=MemoryViews(p)
        if conversation_first and session_id:
            for key,row in list(self.source_descriptors.items()):
                scope,info=self.scopes[key[0]]
                if info['repository']:continue
                from types import SimpleNamespace
                if not views.belongs(scope,SimpleNamespace(**row),session_id):
                    self.source_descriptors.pop(key,None);self.descriptors.pop(key,None)
        plan_start=time.monotonic()
        self.plan=MemoryQueryPlanner().plan(query,policy=self.policy,available_types={r['memory_type'] for r in self.descriptors.values()},
            scope_filters={'workspace_id':workspace,'conversation_id':session_id,'repository_id':repository_id or '',
                           'scope':'conversation' if conversation_first and session_id else 'workspace' if retrieval_scope=='workspace' else 'family',
                           'families':sorted({FAMILIES[i['mode']] for _,i in scopes}), 'authorized_scopes':len(scopes)})
        timings['plan_compute_ms']=(time.monotonic()-plan_start)*1000
        context_dedup='source-family-v1' if strategy in ('C6_evidence','C8_combined') else 'quotes-v2'
        timings['planner_ms']=(time.monotonic()-start)*1000
        warnings=[];groups={m:[] for m in HYBRID};lane_trace={};requests=defaultdict(int)
        exact=self.exact(query) if self.policy.code else []
        exact_valid=[]
        for sk,uid in exact:
            m=self.load(self.scopes[sk][0],[uid]).get(uid)
            if m and self.valid(self.scopes[sk][0],m):exact_valid.append((sk,uid))
        fast=bool(self.policy.cost and self.plan.exact_definition and len(exact_valid)==1)
        vector=sparse=None
        if self.descriptors and not fast:
            try:
                vector,sparse=p.retrieval.encode_query(query);requests.update({'dense_encode':1,'sparse_encode':1})
                timings.update({k:v for k,v in getattr(p.retrieval.local,'timings',{}).items() if isinstance(v,(int,float))})
            except ModelError:warnings.append('Dense / Sparse 查询不可用，使用真实 BM25 结果')
        def recall(scope,ids,methods,depth):
            result={m:[] for m in methods}
            if not ids:return result
            with p.coordinator.search_view(scope) as view:
                for method in methods:
                    t=time.monotonic()
                    if method=='bm25':hits=p.state.lexical.search(scope,query,limit=depth,published_seq=view.published_seq,candidate_uids=ids)
                    elif vector is not None:
                        frontier=p.state._rows('SELECT published_seq,status FROM projection_frontiers WHERE scope=? AND channel=?',(scope.key,p.projection_channel));self.sql_queries+=1
                        if not frontier or frontier[0]['status']!='ready' or frontier[0]['published_seq']!=view.published_seq:
                            warnings.append('部分向量投影未就绪，保留基础检索结果');continue
                        try:
                            hits=p.projection.search(scope,method,vector if method=='cosine' else sparse,limit=depth,published_seq=view.published_seq,candidate_uids=ids)
                            requests[method+'_search']+=1
                        except Exception:
                            warnings.append('部分向量投影不可用，保留基础检索结果');continue
                    else:continue
                    result[method]=[((scope.key,h.logical_uid),h.score) for h in hits]
                    timings[method+'_search_ms']=timings.get(method+'_search_ms',0)+(time.monotonic()-t)*1000
            return result
        t=time.monotonic()
        if not fast:
            for scope,_ in scopes:
                ids=[uid for sk,uid in self.descriptors if sk==scope.key]
                for method,rows in recall(scope,ids,HYBRID,24).items():groups[method].extend(rows)
        timings['candidate_search_ms']=(time.monotonic()-t)*1000
        lane_trace['hybrid']={'methods':list(HYBRID),'candidates':sum(len(v) for v in groups.values()),'ms':round(timings['candidate_search_ms'],3),'executed':not fast}
        # Per-channel ranks, never addition of heterogeneous raw scores.
        def fused(group):
            scores=defaultdict(float)
            for rows in group.values():
                unique=[];seen=set()
                for key,score in sorted(rows,key=lambda x:(-x[1],x[0])):
                    if key in seen:continue
                    seen.add(key);unique.append(key)
                for rank,key in enumerate(unique[:40],1):scores[key]+=1/(60+rank)
            return sorted(scores,key=lambda k:(-scores[k],k)),scores
        ranked,scores=fused(groups)
        for sk in self.scopes:
            self.load(self.scopes[sk][0],[u for k,u in ranked if k==sk])
        def valid_keys(keys):
            return [key for key in keys if (m:=self.load(self.scopes[key[0]][0],[key[1]]).get(key[1])) and self.valid(self.scopes[key[0]][0],m)]
        ranked=valid_keys(ranked);selected=[]
        valid_pool=set(ranked)
        consumed={method:len({key for key,_ in rows if key in valid_pool}) for method,rows in groups.items()}
        if fast:selected=exact_valid
        if self.policy.quotas and not fast:
            extra_queries=0
            for lane in self.plan.selected_lanes:
                t=time.monotonic();pool=[k for k in ranked if self.descriptors[k]['memory_type'] in lane.memory_types]
                if strategy=='adaptive_v33' and lane.methods!=HYBRID:
                    local_rank,_=fused({m:[(k,v) for k,v in groups[m] if k in valid_pool and self.descriptors[k]['memory_type'] in lane.memory_types] for m in lane.methods})
                    pool=local_rank
                if len(pool)<lane.candidate_budget and extra_queries<12:
                    supplements={m:[] for m in lane.methods}
                    for scope,_ in scopes:
                        ids=[uid for (sk,uid),r in self.descriptors.items() if sk==scope.key and r['memory_type'] in lane.memory_types]
                        if not ids:continue
                        if extra_queries+len(lane.methods)>12:break
                        for method,rows in recall(scope,ids,lane.methods,8).items():supplements[method].extend(rows)
                        extra_queries+=len(lane.methods)
                    more,_=fused(supplements);pool=list(dict.fromkeys(pool+valid_keys(more)))
                chosen=pool[:lane.candidate_budget]
                selected.extend(k for k in chosen if k not in selected)
                lane_trace[lane.name]={'pool':len(pool),'selected':len(chosen),'methods':list(lane.methods),'ms':round((time.monotonic()-t)*1000,3)}
        if self.policy.code and not fast:selected=exact_valid[:4]+[k for k in selected if k not in exact_valid[:4]]
        if self.policy.visual and 'ocr' in self.plan.query_intents and not fast:
            ocr_methods=('bm25','splade') if strategy=='adaptive_v33' else ('bm25',)
            ocr,_=fused({method:[(k,v) for k,v in groups[method] if self.descriptors.get(k,{}).get('memory_type')=='visual_ocr' and k in valid_pool] for method in ocr_methods})
            selected=list(dict.fromkeys(ocr[:4]+selected));lane_trace['ocr_lexical']={'selected':len(ocr[:4]),'methods':list(ocr_methods)}
        graph_paths=[];graph_edges=0
        if self.plan.expansion_policy['graph_hops']:
            by_name=defaultdict(list)
            for (sk,uid),row in self.descriptors.items():
                if row['memory_type']!='entity':continue
                name=row['metadata'].get('hlm_item',{}).get('name','')
                if name and name in query:by_name[name].append((sk,uid))
            seeds=[v[0] for v in by_name.values() if len(v)==1]
            t=time.monotonic();expanded,graph_edges,graph_paths=self.graph(valid_keys(seeds))
            timings['graph_ms']=(time.monotonic()-t)*1000
            selected=list(dict.fromkeys(selected[:12]+[k for k in expanded[:4] if k in self.descriptors]));lane_trace['graph']={'seeds':len(seeds),'candidates':len(expanded),'edges':graph_edges,'methods':['stored_edges']}
        if not fast:selected=list(dict.fromkeys(selected+ranked))[:16]
        if fast:lane_trace['exact_symbol']={'methods':['exact_symbol'],'selected':len(selected),'executed':True}
        # All candidates, including exact/graph, share the same BGE stage.
        t=time.monotonic();reranker_used=False;ordered=[(k,scores.get(k,0)) for k in selected]
        if selected and not (fast and len(selected)==1):
            try:
                ranking=p.retrieval.rerank(query,[self.load(self.scopes[k[0]][0],[k[1]])[k[1]].text_content[:5000] for k in selected])
                ordered=[(selected[i],score) for i,score in ranking];reranker_used=True;requests['rerank']=1
            except ModelError:warnings.append('BAAI 重排不可用，采用 RRF 排序')
        timings['rerank_ms']=(time.monotonic()-t)*1000
        snapshots={k:(m.version,digest(m.text_content)) for k in selected if (m:=self.load(self.scopes[k[0]][0],[k[1]]).get(k[1]))}
        # Re-read candidates after network calls; never trust a pre-call cache
        # across source update/delete, attachment replacement or repo rescan.
        t=time.monotonic();items=[];emitted=set();source_count=0
        self.objects.clear();self.validity.clear()
        for (sk,uid),score in ordered:
            scope=self.scopes[sk][0];m=self.load(scope,[uid]).get(uid)
            if not m or (m.version,digest(m.text_content))!=snapshots.get((sk,uid)):
                self.rejected.add((sk,uid));continue
            if not self.valid(scope,m,fresh=True):continue
            originals=self.resolve_sources(scope,m) if self.policy.evidence or not p.eligible(scope,m) else []
            historical=not p.eligible(scope,m)
            targets=originals if historical else [(scope,m)]
            for target_scope,target in targets:
                identity=(target_scope.key,target.logical_uid,target.version)
                if identity in emitted:continue
                item=p.serialize(target_scope,target,detail=True)
                if not item['eligible']:
                    self.rejected.add((target_scope.key,target.logical_uid));continue
                item.update(score=score,scope_key=target_scope.key,temporal_status='historical_source' if historical else time_status(target,self.plan))
                item['query_temporal_view']='historical' if self.plan.historical else 'current'
                if self.policy.evidence:
                    # Missing source links do not confer semantic authority.
                    if target.role=='derived' and not originals:
                        self.rejected.add((sk,uid));continue
                    item['evidence_sources']=[{'id':s.logical_uid,'version':s.version,'expected_version':s.version,
                        'scope_key':ss.key,'text':s.text_content,'role':s.role,'active':True,'session_id':s.session_id,
                        'source_locator':s.metadata.get('source_locator'),'media_ids':s.metadata.get('media_ids',[])} for ss,s in originals]
                    source_count+=len(originals)
                if historical:item['historical_locator_id']=uid
                items.append(item);emitted.add(identity)
                if len(items)>=limit:break
            if len(items)>=limit:break
        timings['provenance_ms']=(time.monotonic()-t)*1000
        timings['total_ms']=(time.monotonic()-start)*1000
        channel_counts={k:len(v) for k,v in groups.items()}
        trace={'planner_version':self.plan.version,'strategy':strategy,'plan':self.plan.public(),
            'profile':p.vector_profile.hash,'collection':p.vector_collection,'sparse_model':p.vector_profile.splade_model,
            'candidate_counts':channel_counts,'rrf_consumed':consumed,'lanes':lane_trace,'rrf_k':60,'rrf_candidates':len(selected),
            'reranker':p.settings.reranker_model,'reranker_used':reranker_used,'reranker_candidates':len(selected) if reranker_used else 0,
            'three_channels_consumed':all(consumed.values()),'retrieval_complete':all(consumed.values()) and reranker_used,
            'invalid_excluded':len(self.rejected),'original_sources':source_count,'source_reads':self.source_reads,'executor_sql_calls':self.sql_queries,
            'expanded_source_unique':len(self.expanded_sources),
            'graph_paths':graph_paths,'fast_exact':fast,'requests':dict(requests),'extra_llm_calls':0,
            'fallbacks':sorted(set(warnings)),'final_evidence_ids':[m['id'] for m in items],
            'timings':{k:round(v,3) for k,v in timings.items()}}
        if strategy=='adaptive_v33':
            trace['execution_path']='unique_verified_symbol' if fast else 'typed_hybrid_with_sources'
            trace['retrieval_complete']=bool(items) if fast else not bool(warnings)
            trace['candidate_duplicates_removed']=sum(channel_counts.values())-len(set(k for values in groups.values() for k,_ in values))
        return {'items':items,'warnings':sorted(set(warnings)),'query':query,'route':{**legacy.public(),
            'conversation_id':session_id,'conversation_first':conversation_first,'searched_scopes':len(scopes),
            'searched_families':sorted({FAMILIES[i['mode']] for _,i in scopes}),
            'searched_repositories':sorted({i['repository'] for _,i in scopes if i['repository']}),'context_dedup':context_dedup,
            'context_semantics':strategy in ('C6_evidence_v3','C8_combined_v3','typed','typed_graph','typed_cost','adaptive_v33'),'memory_query_plan':self.plan.public(),'trace':trace}}
