"""Bounded views of stored canonical objects and explicit provenance only."""
import json
from mandol_substrate.state import Scope
from .store import AppError
from .typed_memory import is_raw
from .memory_quality import validity,group_records,labels,logical_type

PROVENANCE={'SUPPORTED_BY','DERIVED_FROM','SUPPORTS','SOURCE_OF','MENTIONS_SOURCE'}
SEMANTIC={'SUBJECT','OBJECT','RELATION_SUBJECT','RELATION_OBJECT','HAS_PARTICIPANT','PARTICIPATES_IN','CONTRADICTS','SUPERSEDES','UPDATES','RETRACTS'}

def relation_category(relation):
    if relation in PROVENANCE or relation.startswith('SUPPORTS_') or relation in ('CONTEXT_FROM','CONTEXT_FOR'):return 'provenance'
    return 'semantic' if relation in SEMANTIC else 'structure'

class MemoryViews:
    def __init__(self,platform):self.p=platform

    def resolve_reference(self,scope,uid,mapping):
        """Resolve stored navigation aliases, never fuzzy entity identity."""
        m=self.p.state.get_memories(scope,[uid]).get(uid)
        if not m or m.lifecycle=='deleted' or m.commit_seq>self.p.state.scope_state(scope)['published_seq']:return None
        ref=m.metadata.get('hlm_reference')
        if not ref:return m.logical_uid if m.active else None
        # The canonical writer deliberately creates Reference records with
        # active=False so they cannot compete as facts or BM25/vector documents.
        # Deletion and target validity remain independent checks.
        if not m.active and not (m.memory_type=='canonical_reference' and m.memory_level=='Reference'):return None
        target=mapping.get(ref.get('scope_key'))
        if not target:return None
        target_scope=target[0] if isinstance(target,tuple) else target
        if target_scope.user_id!=scope.user_id:return None
        if ref.get('namespace',target_scope.namespace)!=target_scope.namespace or ref.get('user_id',scope.user_id)!=scope.user_id:return None
        actual=self.p.state.get_memories(target_scope,[ref['uid']]).get(ref['uid'])
        from mandol_substrate.state.models import digest
        if not actual or not actual.active or actual.version!=ref['version'] or digest(actual.text_content)!=ref['text_hash'] or actual.commit_seq>self.p.state.scope_state(target_scope)['published_seq']:return None
        return actual.logical_uid

    def alias_ids(self,scope,uids,mapping):
        if not uids:return []
        rows=self.p.state._rows("SELECT logical_uid FROM memories WHERE scope=? AND memory_type IN ('reference','canonical_reference') AND json_extract(metadata,'$.hlm_reference.uid') IN ("+','.join('?' for _ in uids)+') LIMIT 500',[scope.key,*uids])
        return [r['logical_uid'] for r in rows if self.resolve_reference(scope,r['logical_uid'],mapping) in uids]

    def scopes(self,owner,workspace,repository=None):
        return [(s,i) for s,i in self.p.store.scopes(owner,workspace) if not i['repository'] or i['repository']==repository]

    def belongs(self,scope,memory,session):
        if not session:return True
        if memory.memory_type=='reference':return False
        refs=memory.metadata.get('source_versions',{}) or {r['uid']:r['version'] for r in memory.metadata.get('source_refs',[])}
        if refs:
            originals=self.p.state.get_memories(Scope('chinasoft.raw.v1',scope.user_id),list(refs))
            return bool(originals) and len(originals)==len(refs) and all(m.session_id==session for m in originals.values())
        return memory.session_id==session

    def dependents(self,scope_keys,uids):
        """Both persisted provenance formats, same bounded reverse traversal."""
        if not scope_keys or not uids:return []
        marks=','.join('?' for _ in uids)
        return self.p.state._rows('SELECT logical_uid FROM memories WHERE scope IN ('+','.join('?' for _ in scope_keys)+") AND (EXISTS (SELECT 1 FROM json_each(memories.metadata,'$.source_versions') ref WHERE ref.key IN ("+marks+")) OR EXISTS (SELECT 1 FROM json_each(memories.metadata,'$.source_refs') ref WHERE json_extract(ref.value,'$.uid') IN ("+marks+'))) ORDER BY created_at_ms DESC LIMIT 500',[*scope_keys,*uids,*uids])

    def select(self,owner,workspace,session=None,repository=None,kind=None,memory_type=None,offset=0,limit=100,center=None,keyword='',relationship_view='all',state_view='current'):
        if state_view not in ('current','history','all'):raise AppError('无效记忆状态视图')
        self.selection_info={'state_view':state_view,'candidates':0,'scanned':0,'complete':True,'matched_records':0,'authorization':'server_enforced'}
        if session:self.p.store.session(owner,workspace,session)
        if repository:self.p.store.repository(owner,workspace,repository)
        scopes=self.scopes(owner,workspace,repository);mapping={s.key:(s,i) for s,i in scopes}
        if not mapping:
            if center:raise AppError('节点不属于当前范围',404)
            return [],0
        where=['m.scope IN ('+','.join('?' for _ in mapping)+')','m.commit_seq<=s.published_seq',"m.memory_type NOT IN ('reference','canonical_reference')","json_type(m.metadata,'$.hlm_reference') IS NULL",
               "NOT EXISTS (SELECT 1 FROM cs_sessions c WHERE c.id=m.session_id AND c.deleted=1)"]
        args=list(mapping)
        if relationship_view=='semantic':where.append("(m.memory_level<>'L0_Observation' OR m.role='derived')")
        if session:
            # HLM objects may have no session_id; explicit source versions establish membership.
            where.append("(m.session_id=? OR EXISTS (SELECT 1 FROM json_each(m.metadata,'$.source_versions') ref JOIN memories src ON src.logical_uid=ref.key JOIN cs_scopes own ON own.scope=src.scope WHERE src.session_id=? AND own.owner=? AND own.workspace=?) OR EXISTS (SELECT 1 FROM json_each(m.metadata,'$.source_refs') ref JOIN memories src ON src.logical_uid=json_extract(ref.value,'$.uid') JOIN cs_scopes own ON own.scope=src.scope WHERE src.session_id=? AND own.owner=? AND own.workspace=?))")
            args.extend([session,session,owner,workspace,session,owner,workspace])
        if kind:
            if kind not in ('raw','derived'):raise AppError('无效来源分类')
            where.append("(m.memory_level='L0_Observation' AND m.role<>'derived')" if kind=='raw' else "(m.memory_level<>'L0_Observation' OR m.role='derived')")
        if memory_type:
            from .typed_memory import TYPE_MAP
            physical={k for types in TYPE_MAP.values() for k,v in types.items() if v==memory_type}|{memory_type}
            where.append('m.memory_type IN ('+','.join('?' for _ in physical)+')');args.extend(physical)
            if memory_type=='image_text':where.append("COALESCE(json_array_length(m.metadata,'$.media_ids'),0)>0")
            if memory_type=='conversation_message':where.append("COALESCE(json_array_length(m.metadata,'$.media_ids'),0)=0")
        if keyword:
            where.append('instr(m.text_content,?)>0');args.append(keyword)
        if center:
            scope,m,_=self.locate(owner,workspace,center,repository)
            if scope.key not in mapping or not self.belongs(scope,m,session):raise AppError('节点不属于当前范围',404)
            neighbors={center,*(m.metadata.get('source_versions',{}) or {r['uid']:r['version'] for r in m.metadata.get('source_refs',[])})}
            remaining=500
            for linked_scope,_ in mapping.values():
                anchors=[center,*self.alias_ids(linked_scope,{center},mapping)]
                marks=','.join('?' for _ in anchors)
                rows=self.p.state._rows('SELECT source_uid,target_uid FROM graph_edges WHERE scope=? AND '+('active=1 AND ' if state_view=='current' else '')+'commit_seq<=? AND (source_uid IN ('+marks+') OR target_uid IN ('+marks+')) LIMIT ?',[linked_scope.key,self.p.state.scope_state(linked_scope)['published_seq'],*anchors,*anchors,remaining])
                remaining-=len(rows)
                for r in rows:
                    for key in (r['source_uid'],r['target_uid']):
                        direct=self.p.state.get_memories(linked_scope,[key]).get(key)
                        resolved=key if direct and not direct.metadata.get('hlm_reference') else self.resolve_reference(linked_scope,key,mapping)
                        if resolved:neighbors.add(resolved)
                if remaining<=0:break
            # Reverse provenance across typed namespaces, bounded in SQL.
            rows=self.dependents(list(mapping),[center])
            neighbors.update(r['logical_uid'] for r in rows)
            where.append('m.logical_uid IN ('+','.join('?' for _ in neighbors)+')');args.extend(neighbors)
        sql=' FROM memories m JOIN scopes s USING(scope) WHERE '+' AND '.join(where)
        candidates=self.p.state._rows('SELECT COUNT(*) n'+sql,args)[0]['n']
        order=' ORDER BY (m.logical_uid=?) DESC,m.created_at_ms DESC,m.logical_uid' if center else ' ORDER BY m.created_at_ms DESC,m.logical_uid'
        rows=self.p.state._rows('SELECT m.scope,m.logical_uid'+sql+order+' LIMIT 5001',args+([center] if center else []));result=[]
        for r in rows[:5000]:
            scope,info=mapping[r['scope']];m=self.p.state.get_memories(scope,[r['logical_uid']])[r['logical_uid']]
            if not self.belongs(scope,m,session) or not self.p.visible(scope,m):continue
            quality=validity(self.p,scope,m)
            if state_view=='current' and not quality['eligible']:continue
            if state_view=='history' and quality['eligible']:continue
            result.append((scope,m,dict(info,_quality=quality)))
        self.selection_info={'state_view':state_view,'candidates':candidates,'scanned':min(len(rows),5000),'complete':candidates<=5000,'matched_records':len(result),'authorization':'server_enforced'}
        return result[offset:offset+limit],len(result)

    def locate(self,owner,workspace,key,repository=None):
        # Read-only audit access never broadens the retrieval gate. Deleted
        # conversations remain hidden under the existing ownership policy.
        self.p.store.workspace(owner,workspace)
        for scope,info in self.p.store.scopes(owner,workspace,repository_id=repository):
            m=self.p.state.get_memories(scope,[key]).get(key)
            if m and not m.metadata.get('hlm_reference') and m.memory_type not in ('reference','canonical_reference') and self.p.visible(scope,m) and m.commit_seq<=self.p.state.scope_state(scope)['published_seq']:
                return scope,m,info
        raise AppError('记忆不存在或无权访问',404)

    def graph(self,owner,workspace,session=None,repository=None,kind=None,memory_type=None,offset=0,limit=100,center=None,relationship_view='all',keyword='',relation=None,state_view='current',aggregate=False,expand_group=None):
        if relationship_view not in ('all','semantic','provenance'):raise AppError('无效关系视图')
        records,_=self.select(owner,workspace,session,repository,kind,memory_type,0,5000,center,keyword,relationship_view,'all' if state_view=='history' else state_view)
        selection=dict(self.selection_info)
        historical=set()
        if state_view=='history':
            historical={m.logical_uid for _,m,i in records if not i['_quality']['eligible']}
            support={uid for _,m,_ in records if m.logical_uid in historical for uid in (m.metadata.get('source_versions',{}) or {r['uid']:r['version'] for r in m.metadata.get('source_refs',[])})}
            records=[r for r in records if r[1].logical_uid in historical|support or r[1].logical_uid==center]
        groups,member_map=group_records(records,owner,workspace) if aggregate else ({m.logical_uid:[(s,m,i)] for s,m,i in records},{m.logical_uid:m.logical_uid for _,m,_ in records})
        if expand_group in groups:
            expanded=groups[expand_group];new_groups={}
            for gid,items in groups.items():
                if gid==expand_group:
                    for s,m,i in expanded:new_groups[m.logical_uid]=[(s,m,i)];member_map[m.logical_uid]=m.logical_uid
                else:new_groups[gid]=items
            groups=new_groups
        total=len(groups);record_total=len(records)
        page_size=max(1,limit//2) if relationship_view=='provenance' and memory_type and not center else limit
        visible_groups=list(groups.items())[offset:offset+page_size]
        records=[r for _,group in visible_groups for r in group]
        contextual=sum(len(group) for _,group in visible_groups if all(m.logical_uid not in historical for _,m,_ in group)) if state_view=='history' else 0
        if page_size<limit and records:
            related=[];selected_uids={m.logical_uid for _,m,_ in records}
            if any(not is_raw(m) for _,m,_ in records):
                for _,m,_ in records:
                    for uid in m.metadata.get('source_versions',{}) or {r['uid']:r['version'] for r in m.metadata.get('source_refs',[])}:
                        if uid not in related:related.append(uid)
            else:
                allowed={s.key for s,_ in self.scopes(owner,workspace,repository)}
                rows=self.dependents(list(allowed),list(selected_uids))
                related=[r['logical_uid'] for r in rows]
            allowed={s.key for s,_ in self.scopes(owner,workspace,repository)}
            for uid in related:
                if len(visible_groups)>=limit:break
                if uid in selected_uids:continue
                try:source_scope,source,info=self.locate(owner,workspace,uid,repository)
                except AppError:continue
                if source_scope.key not in allowed or not self.belongs(source_scope,source,session):continue
                quality=validity(self.p,source_scope,source)
                if state_view=='current' and not quality['eligible']:continue
                record=(source_scope,source,dict(info,_quality=quality));visible_groups.append((uid,[record]));records.append(record);member_map[uid]=uid;selected_uids.add(uid);contextual+=1
        nodes=[];edges={};uids={m.logical_uid for _,m,_ in records};edge_budget_reached=False
        for gid,group in visible_groups:
            scope,m,info=group[0];item=self.p.serialize(scope,m);label=labels(m,info)
            nodes.append({'id':gid,'label':label['short'],**label,'memory':item,'is_group':len(group)>1,
                          'record_count':len(group),'member_ids':[x.logical_uid for _,x,_ in group[:64]],
                          'members_truncated':len(group)>64,'group_id':gid if len(group)>1 else None})
        for scope,m,_ in records:
            for src,version in (m.metadata.get('source_versions',{}) or {r['uid']:r['version'] for r in m.metadata.get('source_refs',[])}).items():
                if src in uids:
                    eid='provenance:'+m.logical_uid+':'+src
                    edges[eid]={'id':eid,'source':m.logical_uid,'target':src,'relation':'SUPPORTED_BY','category':'provenance','evidence':{'source_version':version,'basis':'stored_source_versions','version_match':any(x.logical_uid==src and x.version==version for _,x,_ in records)}}
        allowed={s.key:s for s,_ in self.scopes(owner,workspace,repository)}
        for scope in allowed.values():
            keys=[m.logical_uid for s,m,_ in records if s.key==scope.key]
            keys+=self.alias_ids(scope,uids,allowed)
            if not keys:continue
            placeholders=','.join('?' for _ in keys)
            seq=self.p.state.scope_state(scope)['published_seq']
            rows=self.p.state._rows('SELECT * FROM graph_edges WHERE scope=? AND '+('active=1 AND ' if state_view=='current' else '')+'commit_seq<=? AND (source_uid IN ('+placeholders+') OR target_uid IN ('+placeholders+')) LIMIT 2000',[scope.key,seq,*keys,*keys])
            edge_budget_reached|=len(rows)>=2000
            def resolve(key):
                if key in uids:return key
                resolved=self.resolve_reference(scope,key,allowed)
                return resolved if resolved in uids else None
            for e in rows:
                source=resolve(e['source_uid']);target=resolve(e['target_uid'])
                if not source or not target:continue
                provenance=relation_category(e['relation_type'])=='provenance'
                if provenance:
                    existing=next((x for x in edges.values() if x['source']==source and x['target']==target and x['category']=='provenance'),None)
                    if existing is not None:
                        existing.setdefault('backing_edge_ids',[]).append(e['edge_id']);continue
                edges[e['edge_id']]={'id':e['edge_id'],'source':source,'target':target,'relation':e['relation_type'],'category':relation_category(e['relation_type']),'evidence':json.loads(e['provenance']),'edge_storage_active':bool(e['active']),'edge_version':e['version']}
        raw_visible=[e for e in edges.values() if (relationship_view=='all' or e['category']==relationship_view or relationship_view=='semantic' and e['category']=='structure') and (not relation or e['relation']==relation)]
        merged={}
        for edge in raw_visible:
            source=member_map.get(edge['source'],edge['source']);target=member_map.get(edge['target'],edge['target'])
            key=(source,target,edge['relation'],edge['category'])
            if key not in merged:merged[key]=dict(edge,source=source,target=target,edge_ids=[],actual_edges=[],record_count=0)
            value=merged[key];backing=edge.get('backing_edge_ids') or [edge['id']];value['record_count']+=len(backing)
            if len(value['edge_ids'])<128:value['edge_ids'].extend(backing[:128-len(value['edge_ids'])]);value['actual_edges'].append(edge)
        if state_view=='current':
            nodes=[n for n in nodes if n['memory']['eligible']]
        node_ids={n['id'] for n in nodes}
        visible=[e for e in merged.values() if e['source'] in node_ids and e['target'] in node_ids]
        return {'nodes':nodes,'edges':visible,'hidden_relations':len(edges)-len(raw_visible),'edge_budget_reached':edge_budget_reached,
                'relationship_view':relationship_view,'state_view':state_view,'aggregation':'conservative-v34' if aggregate else 'none',
                'record_total':record_total,'history_record_total':len(historical) if state_view=='history' else None,'visible_record_count':len(records),'context_nodes_included':contextual,'page_size':page_size,'display_total':total,'selection':selection,
                'total':total,'offset':offset,'limit':limit,'truncated':offset+page_size<total,
                'scope':'conversation' if session else 'conversations','session_id':session,'max_depth':1}

    def members(self,owner,workspace,gid,session=None,repository=None,state_view='current',offset=0,limit=24):
        records,_=self.select(owner,workspace,session,repository,limit=5000,state_view=state_view)
        groups,_=group_records(records,owner,workspace)
        if gid not in groups:raise AppError('显示分组已变化或不属于当前范围',404)
        members=groups[gid]
        return {'id':gid,'items':[self.p.serialize(s,m,detail=True) for s,m,_ in members[offset:offset+limit]],'total':len(members),'offset':offset,'limit':limit,'display_only':True}

    def relations(self,owner,workspace,key,repository=None,state_view='current'):
        _,_,info=self.locate(owner,workspace,key,repository)
        graph=self.graph(owner,workspace,repository=repository or info['repository'] or None,center=key,limit=500,state_view=state_view)
        memories={n['id']:n['memory'] for n in graph['nodes']};items=[]
        for edge in graph['edges']:
            if key not in (edge['source'],edge['target']):continue
            other=edge['target'] if edge['source']==key else edge['source']
            if other in memories:
                items.append({'relation':edge['relation'],'memory':memories[other],'direction':'outgoing' if edge['source']==key else 'incoming',
                              'edge_id':edge['id'],'category':edge['category'],'provenance':edge['evidence']})
        return {'items':items,'state_view':state_view,'truncated':graph['truncated'] or graph['edge_budget_reached']}

    def history(self,owner,workspace,key):
        scope,m,info=self.locate(owner,workspace,key)
        versions={m.version:{'version':m.version,'text':m.text_content,'kind':'canonical_current','active':m.active,'quality':validity(self.p,scope,m)}}
        if is_raw(m):
            rows=self.p.store.rows('SELECT payload,receipt,status,created_at FROM cs_ingestions WHERE owner=? AND workspace=? ORDER BY created_at LIMIT 5000',(owner,workspace))
            for row in rows:
                payload=json.loads(row['payload']);receipt=json.loads(row['receipt'])
                if row['status'] not in ('saved','completed','ready','persisted','indexing','deriving'):continue
                rev=payload.get('revision')
                if rev and rev.get('uid')==key and not rev.get('delete'):
                    v=rev['expected_version']+1
                    if v<=m.version:versions.setdefault(v,{'version':v,'text':rev['content'],'kind':'recorded_revision_input','recorded_at':row['created_at']})
                elif key in receipt.get('memory_ids',[]):
                    index=receipt['memory_ids'].index(key);messages=payload.get('messages',[])
                    if index<len(messages):versions.setdefault(1,{'version':1,'text':messages[index].get('content',''),'kind':'recorded_original_input','recorded_at':row['created_at']})
        return {'id':key,'items':[versions[v] for v in sorted(versions,reverse=True)],'current_version':m.version,
                'complete':len(versions)==m.version,'note':'Only persisted canonical content and reliable ingestion inputs; missing old snapshots are not reconstructed as facts.'}

    def timeline(self,owner,workspace,session,offset=0,limit=30):
        self.p.store.session(owner,workspace,session)
        rows=self.p.store.rows('SELECT * FROM cs_ingestions WHERE owner=? AND workspace=? AND session=? ORDER BY created_at DESC,id DESC LIMIT ? OFFSET ?',(owner,workspace,session,limit,offset));items=[]
        for r in rows:
            receipt=json.loads(r['receipt']);payload=json.loads(r['payload']);sources=[]
            for key in receipt['memory_ids']:
                try:
                    scope,m,_=self.locate(owner,workspace,key);item=self.p.serialize(scope,m)
                    index=receipt['memory_ids'].index(key);originals=payload.get('messages',[])
                    if index<len(originals):item.update(recorded_text=originals[index].get('content',''),recorded_version=1,current_version=m.version)
                    sources.append(item)
                except AppError:pass
            jobs=self.p.store.rows('SELECT id,kind,status,progress,error FROM cs_jobs WHERE owner=? AND workspace=? AND ingestion=?',(owner,workspace,r['id']))
            for j in jobs:j['progress']=json.loads(j['progress'])
            chat=self.p.store.rows('SELECT id,answer,status,updated_at,error FROM cs_chat WHERE owner=? AND workspace=? AND session=? AND request_id=?',(owner,workspace,session,payload['request_id'].removeprefix('chat-memory-'))) if payload['request_id'].startswith('chat-memory-') else []
            items.append({'id':r['id'],'created_at':r['created_at'],'status':r['status'],'routes':receipt.get('routes',[]),'sources':sources,'jobs':jobs,'answer':chat[0] if chat else None})
        total=self.p.store.rows('SELECT COUNT(*) n FROM cs_ingestions WHERE owner=? AND workspace=? AND session=?',(owner,workspace,session))[0]['n']
        return {'items':items,'total':total,'offset':offset,'limit':limit}
