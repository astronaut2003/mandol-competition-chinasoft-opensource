from dataclasses import replace, asdict, fields
from types import SimpleNamespace
from pathlib import Path
from threading import Thread, Event, Lock
import hashlib, json, time, os, io, logging, re
from PIL import Image, UnidentifiedImageError
from pydantic import Field
from mandol_substrate.state import SQLiteStateStore, RetrievalProfile, CanonicalMemory, Mutation, Edge, Scope
from mandol_substrate.state.models import digest, canonical_json, StateUnavailable
from mandol_substrate.text import MandolTextProcessor
from mandol_substrate.contracts import ResourceIdentity, PublicationPolicy
from mandol_substrate.publication import PublicationCoordinator
from mandol_substrate.vector import MinimalProjectionCodec
from mandol_substrate.qdrant.backend import QdrantProjectionBackend
from mandol_substrate.qdrant.policy import QdrantProjectionPolicy
from mandol_substrate.representation import MemoryUnit
from mandol_substrate.retrieval.results import MultiRetrievalResults, RetrievalResult, RetrievalMethod
from mandol_substrate.retrieval.fusion import ScoreFusion
from mandol_competition.textual.core.hlm_production import CanonicalHighLevelMemory, ProductionHLMRecipe, HighLevelMemoryPolicy
from mandol_competition.textual.core.heterogeneous_worker import typed_scope
from mandol_competition.code.experience import ExperienceBuilder
from mandol_competition.multimodal.episodes import prepare_navigation
from .config import Settings, ROOT
from .store import AppStore, AppError, now, uid
from .models import GenerationClient, RetrievalModels, HLMProvider, ModelError
from .schemas import Strict
from .highlevel import ChinaSoftHighLevelMemory, ChinaSoftHLMRecipe, ConversationRecipeV3, ConversationMemoryV3
from .typed_memory import FAMILIES, MODES, TYPE_MAP, view_type, is_raw, plan_query
from .repository import RepositoryManager

logger = logging.getLogger(__name__)


class VisionResult(Strict):
    description: str = Field(min_length=1, max_length=4000)
    visible_text: list[str] = Field(max_length=40)
    objects: list[str] = Field(max_length=30)
    uncertainties: list[str] = Field(max_length=20)


class DeferredProjection:
    """API/L0 can start while Qdrant is down; only projection work needs it."""
    def __init__(self, factory): self.factory, self.backend, self.lock = factory, None, Lock()

    def get(self):
        with self.lock:
            if self.backend is None: self.backend = self.factory()
            return self.backend

    def __getattr__(self, key): return getattr(self.get(), key)

    def close(self):
        if self.backend: self.backend.close()


class Platform:
    def __init__(self, settings=None, *, generation_transport=None, retrieval_transport=None,web_transport=None):
        self.settings = s = settings or Settings(); s.data_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(s.data_dir,0o700)
        import jieba
        jieba.dt.tmp_dir=str(s.data_dir)
        self.processor = MandolTextProcessor(use_jieba=True)
        self.profile = RetrievalProfile(s.embedding_model, 'provider-declared-v1', s.embedding_dimension,
            self.processor.fingerprint, 'chinasoft-lexical-sparse',
            'sha256-token-tf-v1',
            self.processor.fingerprint, options={'deployment':'chinasoft-single-machine-v1', 'embedding_endpoint':s.embedding_url,
                                               'reranker':s.reranker_model, 'sparse_endpoint':''})
        from .projection import canonical_profile
        self.profile=canonical_profile(s,self.profile)
        identity = ResourceIdentity('chinasoft', 'chinasoft.raw.v1', s.collection)
        self.state = SQLiteStateStore(s.data_dir / 'state.sqlite3', self.profile, processor=self.processor, identity=identity)
        self.store = AppStore(self.state, s)
        from .auth import Accounts
        self.accounts=Accounts(self.store)
        if s.retrieval_backend == 'vllm':
            from .native_retrieval import NativeRetrievalModels
            self.retrieval = NativeRetrievalModels(s, self.processor, retrieval_transport)
        else:
            self.retrieval = RetrievalModels(s, self.processor, retrieval_transport)
        self.generation = GenerationClient(s, self.store, generation_transport)
        from .web_tools import WebTools
        self.web = WebTools(s,self.store,web_transport)
        policy = QdrantProjectionPolicy()
        self.projection = DeferredProjection(lambda: QdrantProjectionBackend(s.qdrant_url, self.profile,
            state=self.state, storage_id=self.state.db_uuid, identity=identity, collection_name=s.collection, optional_projection=True))
        self.coordinator = PublicationCoordinator(self.state, self.projection, MinimalProjectionCodec(self.profile, policy.hash), self.retrieval,
            policy=PublicationPolicy('chinasoft-canonical-lexical-async-vector-v1', required_projection=False, encode_before_prepare=False),
            retries=2, verify_timeout=25, batch_size=24)
        self.projection_channel='dense';self.vector_profile=self.profile;self.vector_collection=s.collection;self.versioned_projection=None
        if s.sparse_url:
            from .projection import VersionedProjection
            self.versioned_projection=VersionedProjection(self,DeferredProjection)
            self.legacy_projection=self.projection;self.projection=self.versioned_projection.backend
            self.projection_channel=self.versioned_projection.channel;self.vector_profile=self.versioned_projection.profile;self.vector_collection=self.versioned_projection.collection
        self.recipe = ProductionHLMRecipe(self.profile.hash, model=s.llm_model,
            provider='deepseek-official' if 'api.deepseek.com' in s.llm_base_url else 'openai-compatible', actual_model=s.llm_request_model,
            version='chinasoft-four-tower-v1-' + s.generation_fingerprint,
            policy=HighLevelMemoryPolicy(hierarchical_enabled=True, episodic_enabled=True, entity_relation_enabled=True,
                                        profile_enabled=True, navigation_enabled=True, semantic_evidence_enabled=True))
        self.hlm = ChinaSoftHighLevelMemory(self.coordinator, self.recipe, HLMProvider(self.generation))
        self.legacy_hlm=self.hlm
        self.recipe=ChinaSoftHLMRecipe(self.profile.hash,model=s.llm_model,provider=self.recipe.provider,actual_model=s.llm_request_model,
            version='chinasoft-scenarios-v2-'+s.generation_fingerprint,policy=self.recipe.policy)
        self.hlm=ChinaSoftHighLevelMemory(self.coordinator,self.recipe,HLMProvider(self.generation),publication_writer=self.legacy_hlm._writer)
        self.v3_recipe=ConversationRecipeV3(**{f.name:getattr(self.recipe,f.name) for f in fields(self.recipe)})
        self.v3_hlm=ConversationMemoryV3(self.coordinator,self.v3_recipe,HLMProvider(self.generation,version3=True),publication_writer=self.legacy_hlm._writer)
        self.repositories = RepositoryManager(self)
        from .memory_clear import MemoryClearing
        self.memory_clearing = MemoryClearing(self)
        self.stop_event, self.threads = Event(), []
        self.media_lock=Lock()
        self.started = now()

    def start(self):
        health=self.retrieval.probe_all(force=True) if self.settings.retrieval_backend=='vllm' else {'sparse':self.retrieval.probe_sparse(force=True)}
        if not all(h['ready'] for h in health.values()) and not self.settings.allow_retrieval_degraded:
            missing=','.join(c for c,h in health.items() if not h['ready'])
            raise ModelError('retrieval_required' if self.settings.retrieval_backend=='vllm' else 'splade_required','检索服务未就绪或配方不兼容：'+missing+'；启动独立 vLLM 服务并检查 diagnose，或显式允许降级')
        self.store.recover(); self.coordinator.recover()
        self.memory_clearing.recover()
        if self.versioned_projection:self.versioned_projection.schedule()
        for kind in ('ingest', 'repository', 'index', 'derive'):
            thread = Thread(target=self.worker, args=(kind,), name='chinasoft-' + kind, daemon=True)
            thread.start(); self.threads.append(thread)

    def close(self):
        self.stop_event.set()
        for thread in self.threads: thread.join(self.settings.llm_timeout + 25)
        if any(t.is_alive() for t in self.threads):
            # Do not close SQLite under a live writer. Process exit and durable leases recover safely.
            logger.error('Background worker shutdown timed out; durable lease will recover'); return
        if hasattr(self,'legacy_projection'):self.legacy_projection.close()
        self.v3_hlm.close();self.hlm.close();self.legacy_hlm.close(); self.generation.close();self.web.close(); self.retrieval.close(); self.projection.close(); self.state.close()

    def worker(self, kind):
        lease = uid('worker_')
        while not self.stop_event.is_set():
            job = self.store.claim(kind, lease)
            if not job:
                if kind=='index' and self.versioned_projection:self.versioned_projection.schedule()
                self.stop_event.wait(0.3); continue
            self.generation.local.job_id = job['id']; started = time.monotonic()
            if kind == 'derive': self.generation.active_build_job = job['id']
            try:
                result = getattr(self, 'do_' + kind)(job)
                if kind=='repository':
                    final=self.store.rows('SELECT status,error_code FROM cs_jobs WHERE id=?',(job['id'],))
                    if final and (final[0]['status']=='cancelled' or final[0]['error_code']=='cancel_requested'):
                        raise AppError('仓库任务已取消，原始来源保留',409,code='repository_cancelled')
                result['duration_ms'] = round((time.monotonic() - started) * 1000, 2)
                self.store.finish(job, result); self.store.metric('job_' + kind, result['duration_ms'])
            except Exception as exc:
                if kind=='repository':
                    cancellation=self.store.rows('SELECT status,error_code FROM cs_jobs WHERE id=?',(job['id'],))
                    if cancellation and (cancellation[0]['status']=='cancelled' or cancellation[0]['error_code']=='cancel_requested'):
                        exc=AppError('仓库任务已按请求取消',409,code='repository_cancelled')
                if kind=='repository' and getattr(exc,'code',None)=='repository_cancelled':
                    with self.state.transaction() as db:
                        db.execute("UPDATE cs_jobs SET status='cancelled',lease=NULL,error_code='repository_cancelled',error=NULL,updated_at=? WHERE id=? AND lease=?",(now(),job['id'],job['lease']))
                        db.execute("UPDATE cs_repositories SET status='cancelled',error=NULL WHERE id=? AND owner=? AND deleted=0",(job['payload']['repository_id'],job['owner']))
                    rows=self.store.rows('SELECT * FROM cs_repositories WHERE id=?',(job['payload']['repository_id'],))
                    if rows and rows[0].get('source_kind')=='remote' and not rows[0]['deleted']:self.repositories.remote.cleanup(rows[0])
                    continue
                if kind=='derive' and getattr(self.generation.local,'source_versions',None):
                    versions=self.generation.local.source_versions
                    current=self.state.get_memories(self.generation.local.source_scope,list(versions))
                    if any(u not in current or not current[u].active or current[u].version!=v for u,v in versions.items()):
                        self.store.cancel_superseded(job);continue
                logger.exception('Background job %s failed: %s', job['id'], type(exc).__name__)
                if isinstance(exc, StateUnavailable): exc = ModelError('publication_unavailable', '检索发布暂时不可用，原始输入已保留', retryable=True)
                if kind=='repository':
                    if getattr(exc,'code',None)=='repository_interrupted':exc=ModelError('repository_interrupted','扫描中断，已完成的来源与进度保留',retryable=True)
                    if getattr(exc,'code',None) in ('git_connection','git_dns_failed','git_timeout','git_http_429','git_http_500','git_http_502','git_http_503','git_http_504'):
                        exc=ModelError(exc.code,exc.message,retryable=True)
                    with self.state.transaction() as db:
                        db.execute("UPDATE cs_repositories SET status='failed',error=? WHERE id=? AND owner=? AND workspace=?",(getattr(exc,'message','仓库处理失败，请查看后台任务'),job['payload']['repository_id'],job['owner'],job['workspace']))
                if kind == 'derive':
                    try:
                        scopes = self.store.scopes(job['owner'],job['workspace'])
                        self.store.enqueue(job['owner'],job['workspace'],'index',{'scopes':[asdict(scope) for scope,_ in scopes]},[job['id'],job['attempts'],'failure-frontier'])
                    except Exception: pass
                self.store.fail(job, exc)
            finally:
                self.generation.local.job_id = None
                self.generation.local.source_versions=None
                if kind == 'derive': self.generation.active_build_job = None

    def raw_scope(self, ingestion):
        data=ingestion.get('payload',{})
        selected=self.store.session(ingestion['owner'],ingestion['workspace'],ingestion['session'],ingestion['mode'])
        repository=data['revision'].get('repository_id',selected['repository']) if data.get('revision') else '' if data.get('unified_raw') else data.get('repository_id') or selected['repository']
        return self.store.scope(ingestion['owner'], ingestion['workspace'], ingestion['mode'],repository)

    def do_repository(self,job):
        return self.repositories.scan(job)

    def do_ingest(self, job):
        ing = self.store.ingestion(job['owner'], job['payload']['ingestion']); data = ing['payload']; scope = self.raw_scope(ing)
        memories = []
        if data.get('revision'):
            rev = data['revision']; old = self.state.get_memories(scope, [rev['uid']]).get(rev['uid'])
            if not old: raise AppError('基础记忆已不存在', 404)
            previous = self.state.request(scope, data['request_id'])
            if previous: self.coordinator.apply(scope, data['request_id'], ing['digest'], Mutation(), {})
            else:
                if old.version != rev['expected_version']: raise AppError('记忆版本冲突', 409)
                if rev.get('delete'):
                    mutation = Mutation(delete_uids=(old.logical_uid,))
                else:
                    raw = dict(old.raw_data, text_content=rev['content']); meta = dict(old.metadata)
                    if meta.get('routing_recipe'):
                        from .routing import routes
                        meta['routes']=routes({'content':rev['content'],'role':old.role,'media_ids':meta.get('media_ids',[]),'attachment_id':meta.get('attachment_id'),'attachment_kind':meta.get('attachment_kind')})
                    raw['content_parts'] = [dict(part, text=rev['content']) if part.get('type')=='text' else part for part in old.raw_data.get('content_parts',[])]
                    memories = [replace(old, version=old.version+1, text_content=rev['content'], raw_data=raw, metadata=meta)]
                    mutation = Mutation(memories=tuple(memories))
                self.coordinator.apply(scope, data['request_id'], ing['digest'], mutation, {'memory_ids':[old.logical_uid]})
        else:
            for i, message in enumerate(data['messages']):
                mid = ing['receipt']['memory_ids'][i]
                memories.append(CanonicalMemory(mid, message['content'] or '[用户上传的图片；视觉理解待后台完成]',
                    raw_data={'text_content':message['content'], 'content_parts':[{'type':'text','text':message['content']}] +
                              [{'type':'image','media_id':m} for m in message['media_ids']]},
                    metadata={'user_id':scope.user_id, 'owner_id':job['owner'], 'workspace_id':ing['workspace'], 'mode':ing['mode'],
                              'memory_family':FAMILIES[ing['mode']], 'repository_id':data.get('repository_id') or '',
                              'routes':message.get('routes',[]),'routing_recipe':data.get('routing_recipe'),
                              'attachment_id':message.get('attachment_id'),'attachment_name':message.get('attachment_name'),'attachment_kind':message.get('attachment_kind'),
                              'attachment_version':message.get('attachment_version',1),'attachment_hash':message.get('attachment_hash'),'attachment_parser':message.get('attachment_parser'),
                              'source_locator':({'attachment_id':message['attachment_id'],'path':message['attachment_name'],'start_line':1,'end_line':len(message['content'].splitlines()) or 1,'file_hash':message.get('attachment_hash'),'repository_id':''} if message.get('attachment_id') else None),
                              'sequence':ing['created_at']*100+i, 'media_ids':message['media_ids'], 'speaker':'当前用户' if message['role']=='user' else '助手', 'aml':True},
                    session_id=ing['session'], request_id=data['request_id'], turn_index=i, role=message['role'],
                    memory_type='source_attachment' if message.get('attachment_id') else 'engineering_raw' if ing['mode']=='code' else 'observation', business_timestamp_ms=ing['created_at']))
            self.coordinator.apply(scope, data['request_id'], ing['digest'], Mutation(memories=tuple(memories)), {'memory_ids':ing['receipt']['memory_ids']})
        with self.state.transaction() as db:
            db.execute("UPDATE cs_ingestions SET status='saved',saved_at=? WHERE id=?", (now(), ing['id']))
        self.store.metric('ingestion_to_L0', now()-ing['created_at'])
        return {'stage':'基础记忆已保存', 'memory_ids':ing['receipt']['memory_ids'], 'saved_at':now(), 'lexical_ready':True}

    def do_index(self, job):
        recipe=job['payload'].get('projection_profile')
        if recipe and recipe!=self.vector_profile.hash:
            progress={'stage':'旧检索 Profile 补建任务已被新版本替代','superseded_profile':recipe,'active_profile':self.vector_profile.hash}
            with self.state.transaction() as db:
                db.execute("UPDATE cs_jobs SET status='cancelled',lease=NULL,error_code='projection_profile_superseded',progress=?,updated_at=? WHERE id=? AND lease=?",(canonical_json(progress),now(),job['id'],job['lease']))
            return progress
        if job['payload'].get('scopes'):
            scopes = [Scope(x['namespace'], x['user_id']) for x in job['payload']['scopes']]
        else:
            ing = self.store.ingestion(job['owner'], job['payload']['ingestion']); scopes = [self.raw_scope(ing)]
            if ing['payload'].get('unified_raw'):
                # Read compatibility for pre-V3.2 jobs: a linked knowledge source
                # was incorrectly treated as the raw conversation's index scope.
                # Derive the boundary from the authorized durable ingestion,
                # never from a client-selected Scope or a foreign repository.
                job={**job,'payload':{**job['payload'],'associated_repository_id':ing['payload'].get('repository_id') or '', 'repository_id':''}}
        for scope in scopes:
            registered=self.store.rows('SELECT * FROM cs_scopes WHERE scope=? AND owner=? AND workspace=?',(scope.key,job['owner'],job['workspace']))
            if not registered:raise AppError('索引任务不能访问其他工作区 Scope',403,code='scope_rejected')
            if job['payload'].get('repository_id') and registered[0]['repository']!=job['payload']['repository_id']:
                raise AppError('索引任务 Repository 不匹配',403,code='scope_rejected')
            frontier=self.state._rows('SELECT published_seq,status FROM projection_frontiers WHERE scope=? AND channel=?',(scope.key,self.projection_channel))
            if frontier and frontier[0]['status']=='ready' and frontier[0]['published_seq']==self.state.scope_state(scope)['published_seq']:continue
            count=self.state._rows('SELECT COUNT(*) n FROM memories WHERE scope=? AND active=1',(scope.key,))[0]['n']
            if not (self.versioned_projection.project(scope,job) if self.versioned_projection else self.project_bounded(scope,job) if count>96 else self.coordinator.project_optional(scope)):
                raise ModelError('projection_unavailable', 'Qdrant / Dense / SPLADE 投影暂未完成；基础词法检索仍可使用', retryable=True)
        if job['payload'].get('repository_id'):
            with self.state.transaction() as db:db.execute("UPDATE cs_repositories SET status='ready' WHERE id=? AND owner=? AND workspace=? AND status='lexical_ready'",(job['payload']['repository_id'],job['owner'],job['workspace']))
        return {'stage':'Dense / Sparse 检索投影已发布', 'scopes':len(scopes), 'ready_at':now()}

    def project_bounded(self,scope,job):
        """Yield existing optional projection between batches; one verified frontier.

        Reuses the coordinator codec/reconciler and canonical gates. A source
        change aborts the snapshot, so a partial corpus never claims readiness.
        """
        c=self.coordinator
        try:
            with c.gates.lease(scope,write=True):
                current=self.state.scope_state(scope);seq=current['published_seq']
                if seq!=current['canonical_seq'] or seq!=current['lexical_seq']:return False
                self.projection.ensure_schema(scope,create=True)
            done=0
            for batch in self.state.iter_memories(scope,c.batch_size):
                if self.stop_event.wait(.03):return False
                with c.gates.lease(scope,write=True):
                    if self.state.scope_state(scope)['published_seq']!=seq:return False
                    c._reconcile_batch(scope,seq,[{'logical_uid':m.logical_uid,'operation':'upsert'} for m in batch],{})
                done+=len(batch)
                with self.state.transaction() as db:db.execute('UPDATE cs_jobs SET progress=?,updated_at=? WHERE id=? AND lease=?',(canonical_json({'stage':'分批构建检索投影','indexed':done,'snapshot_seq':seq}),now(),job['id'],job['lease']))
            with c.gates.lease(scope,write=True):
                if self.state.scope_state(scope)['published_seq']!=seq:return False
                if hasattr(self.projection,'audit_rows'):self.projection.audit_rows(scope)
                self.state.set_projection_frontier(scope,'dense',seq)
            return True
        except Exception as exc:
            self.state.mark_projection_degraded(scope,'dense',type(exc).__name__);return False

    def session_sources(self, scope, session):
        rows = self.state._rows("SELECT logical_uid FROM memories WHERE scope=? AND session_id=? AND active=1 AND memory_level='L0_Observation' ORDER BY created_at_ms,turn_index,logical_uid", (scope.key, session))
        sources = self.state.get_memories(scope, [r['logical_uid'] for r in rows])
        ordered = sorted(sources.values(), key=lambda m:(m.metadata.get('sequence',m.created_at_ms),m.turn_index,m.logical_uid))
        # Explicit recent-window selection; every original remains in L0.
        chosen, budget = [], 0
        for m in reversed(ordered):
            if len(chosen) >= 48 or budget + len(m.text_content) > 80000: break
            chosen.append(m); budget += len(m.text_content)
        return list(reversed(chosen)), len(ordered)-len(chosen)

    def do_derive(self, job):
        ing = self.store.ingestion(job['owner'], job['payload']['ingestion']); scope = self.raw_scope(ing)
        sources, omitted = self.session_sources(scope, ing['session']); scopes = [scope]
        self.generation.local.source_versions={m.logical_uid:m.version for m in sources};self.generation.local.source_scope=scope
        if not sources: return {'generated':0,'stage':'无有效原始来源，无需构建'}
        route=job['payload'].get('route')
        if route:
            sources=[m for m in sources if route in m.metadata.get('routes',[]) and (route!='conversation' or (m.role=='user' and not m.metadata.get('attachment_id')))]
            if not sources:return {'generated':0,'stage':'无适格来源'}
        mode = {'conversation':'textual','engineering':'code','visual':'multimodal'}.get(route,ing['mode']); outputs = []; errors = {}
        if route=='engineering':
            sources=[m for m in sources if (m.metadata.get('repository_id') or '')==(ing['payload'].get('repository_id') or '')]
            if not sources:return {'generated':0,'stage':'当前仓库没有适格工程来源'}
        if route=='source_code':return self.derive_attachment_code(scope,sources,ing,job)
        if mode == 'textual':
            producer=self.v3_hlm if route else self.hlm
            result = producer.build(scope, [m.logical_uid for m in sources])
            errors = result.get('errors', {})
            for kind, ids in result.get('output_uids', {}).items():
                partition = typed_scope(scope, kind, producer.recipe); scopes.append(partition)
                self.store.register_scope(partition, job['owner'], ing['workspace'], mode, kind); outputs.extend(ids)
        elif mode == 'code':
            plan = {**self.settings.contract(), 'generation_fingerprint':self.settings.generation_fingerprint}
            proxy = SimpleNamespace(state=self.state, scope=lambda user:scope)
            def generate(prepared, request_id):
                from .engineering_schema import ENGINEERING_SCHEMA
                messages = [{'role':'system','content':prepared['instruction']}, {'role':'user','content':canonical_json({k:v for k,v in prepared.items() if k!='instruction'})}]
                rec = self.generation.complete(messages, schema=ENGINEERING_SCHEMA)
                return json.loads(rec['content']), {**rec, 'recipe_fingerprint':digest(plan)}
            builder = ExperienceBuilder(SimpleNamespace(store=proxy), generator=generate, model_plan=plan)
            experience_uid='engineering_experience:'+digest([scope.key,{m.logical_uid:m.version for m in sources},builder.fingerprint])
            previous=self.state.get_memories(scope,[experience_uid]).get(experience_uid)
            if previous and self.eligible(scope,previous):
                return {'generated':0,'reused_memory_ids':[experience_uid],'stage':'同一来源版本与配方的工程经验已发布，复用持久结果'}
            mutation = builder.prepare(scope, ing['session'], job['id'], digest([job['id'],{m.logical_uid:m.version for m in sources}]), [replace(m,memory_type='engineering_raw') for m in sources] if route else sources)
            preparation = self.state.ledger.reserve(scope,job['id'],digest([job['id'],{m.logical_uid:m.version for m in sources}])).get('code_experience',{})
            if preparation.get('state') in ('failed','inflight'):
                unknown=bool(self.store.rows("SELECT 1 FROM cs_calls WHERE job_id=? AND state='unknown' LIMIT 1",(job['id'],)))
                raise ModelError('derivation_rejected','工程提炼未完成，原始记录与已有有效经验保留',unknown=unknown)
            updated = []; knowledge = [];association_edges=[]
            for m in mutation.memories:
                m = replace(m, memory_level='L1_Episode', metadata=dict(m.metadata, user_id=scope.user_id, mode=mode, memory_family='code',repository_id=ing['payload'].get('repository_id') or '', generation=plan,source_hash_schema='engineering-source-digest-v1'))
                if ing['payload'].get('repository_id'):
                    associations=[]
                    files=self.store.rows("SELECT path,hash,uids FROM cs_repository_files WHERE repository=? AND status='indexed'",(ing['payload']['repository_id'],))
                    for file in files:
                        mentions=[src.logical_uid for src in sources if re.search(r'(?<![\w/])'+re.escape(file['path'])+r'(?![\w/])',src.text_content)]
                        if not mentions:continue
                        target=json.loads(file['uids'])[0]
                        associations.append({'path':file['path'],'file_hash':file['hash'],'memory_id':target,'basis':'EXPLICIT_PATH_MENTION','source_uids':mentions})
                        association_edges.append(Edge('code-association:'+digest([m.logical_uid,target]),m.logical_uid,target,'RELATED_CODE',provenance={'basis':'EXPLICIT_PATH_MENTION','source_uids':mentions}))
                    m=replace(m,metadata=dict(m.metadata,code_associations=associations[:24]))
                updated.append(m); outputs.append(m.logical_uid)
                for item in m.raw_data['audit'].get('accepted', []):
                    if item['field'] != 'knowledge': continue
                    evidence = [{'source_uid':c['source_uid'], 'quote':self.state.get_memories(scope,[c['source_uid']])[c['source_uid']].text_content[c['start']:c['end']]} for c in item['citations']]
                    kid = 'engineering_knowledge:' + digest([m.logical_uid, item])
                    knowledge.append(CanonicalMemory(kid, item['text'], raw_data={'knowledge_kind':item['kind'], 'evidence':evidence},
                        metadata=dict(m.metadata, evidence=evidence, verification='INFERRED', parent_experience=m.logical_uid),
                        session_id=ing['session'], request_id=job['id'], role='derived', memory_tower='engineering', memory_level='L2_Knowledge', memory_type='engineering_knowledge', spaces=('code_raw',)))
                    outputs.append(kid)
            # Prior knowledge is retired only when a new accepted experience exists.
            old_knowledge = self.state._rows("SELECT logical_uid FROM memories WHERE scope=? AND session_id=? AND active=1 AND memory_type='engineering_knowledge'", (scope.key, ing['session'])) if updated else []
            # Keep earlier experience as historical evidence; source revisions still invalidate it.
            mutation = replace(mutation, memories=tuple(updated+knowledge), spaces=('code_raw',), delete_uids=(),edges=mutation.edges+tuple(association_edges[:24]))
            self.coordinator.apply(scope, job['id']+':publish', digest(asdict(mutation)), mutation, {'outputs':outputs})
            receipt = self.state.ledger.reserve(scope, job['id'], digest([job['id'],{m.logical_uid:m.version for m in sources}])).get('code_experience', {})
            if receipt.get('state') in ('failed','inflight'): errors['experience'] = receipt.get('error_class', 'unknown_remote')
            if receipt.get('audit',{}).get('rejected'): errors['experience_audit'] = 'source_claim_rejected'
        else:
            mutation_memories, edges = [], []
            current_ids = set(ing['receipt']['memory_ids'])
            for source in sources:
                if source.logical_uid not in current_ids: continue
                mids = source.metadata.get('media_ids', [])
                if not mids: continue
                media = [self.store.media(job['owner'], m, ing['workspace'], ing['session']) for m in mids]
                msg = [{'role':'system','content':'描述图片直接可见内容。文字说明只是用户提供的背景，不证明图像内容。不要推断人物身份、图片之间的因果或事件。保留不确定性，返回 JSON。'},
                       {'role':'user','content':[{'type':'text','text':'请提取图片描述、可见文字、物体和不确定项。用户说明：'+source.text_content}] + [self.generation.image_block(m) for m in media]}]
                visual_recipe=digest(['visual-scenario-v2',msg[0],VisionResult.model_json_schema(),self.settings.generation_fingerprint])
                rec = self.generation.complete(msg, vision=True, schema=VisionResult.model_json_schema())
                result = VisionResult.model_validate_json(rec['content']); key = 'visual_memory:' + digest([source.logical_uid,source.version,visual_recipe])
                text = result.description + '\n可见文字：' + '；'.join(result.visible_text) + '\n不确定项：' + '；'.join(result.uncertainties)
                mutation_memories.append(CanonicalMemory(key, text, raw_data=result.model_dump(),
                    metadata={'user_id':scope.user_id,'source_versions':{source.logical_uid:source.version},'source_hashes':{source.logical_uid:hashlib.sha256(source.text_content.encode()).hexdigest()},
                              'media_ids':mids,'generation':rec['model_contract'],'recipe_fingerprint':visual_recipe, 'memory_family':'multimodal','verification':'MODEL_OBSERVATION'},
                    session_id=ing['session'], request_id=job['id'], role='derived', memory_type='visual_memory', memory_level='L1_Visual'))
                edges.append(Edge('visual_support:' + digest([key,source.logical_uid]),key,source.logical_uid,'SUPPORTED_BY', provenance={'source_version':source.version,'media_ids':mids}))
                outputs.append(key)
                for kind,items in (('visual_ocr',result.visible_text),('visual_object',result.objects)):
                    if not items:continue
                    child='visual_'+digest([source.logical_uid,source.version,visual_recipe,kind,items])
                    mutation_memories.append(CanonicalMemory(child,'\n'.join(items),raw_data={'items':items,'uncertainties':result.uncertainties,'observation_kind':kind},
                       metadata={'user_id':scope.user_id,'source_versions':{source.logical_uid:source.version},'media_ids':mids,'generation':rec['model_contract'],'recipe_fingerprint':visual_recipe,
                                 'memory_family':'multimodal','verification':'MODEL_OCR' if kind=='visual_ocr' else 'MODEL_OBJECT_OBSERVATION'},
                       session_id=ing['session'],request_id=job['id'],role='derived',memory_type=kind,memory_level='L1_Visual'))
                    edges.append(Edge('visual_support:'+digest([child,source.logical_uid]),child,source.logical_uid,'SUPPORTED_BY',provenance={'source_version':source.version,'media_ids':mids}));outputs.append(child)
            nav = prepare_navigation(self.state, scope, SimpleNamespace(session_id=ing['session'],user_id=scope.user_id,request_id=job['id']), tuple(m for m in sources if m.logical_uid in current_ids))
            # Existing navigation includes raw records; publish only the derived window.
            visual_nav = tuple(m for m in nav.memories if m.memory_type=='visual_episode')
            self.coordinator.apply(scope, job['id']+':visual', digest([outputs,[m.logical_uid for m in visual_nav]]),
                Mutation(memories=tuple(mutation_memories)+visual_nav, spaces=nav.spaces, edges=tuple(edges)+nav.edges, jobs=nav.jobs, delete_uids=nav.delete_uids), {'outputs':outputs})
        if scopes:
            self.store.enqueue(job['owner'], ing['workspace'], 'index', {'scopes':[asdict(s) for s in scopes]}, [job['id'],outputs,'post-derive'])
        if errors and route=='conversation' and outputs and not self.store.rows("SELECT 1 FROM cs_calls WHERE job_id=? AND state IN ('unknown','failed') LIMIT 1",(job['id'],)):
            return {'generated':len(outputs),'memory_ids':outputs,'warnings':errors,'stage':'审核完成；不适格候选已拒绝，仅发布通过审核的记忆','omitted_window_sources':omitted}
        if errors:
            with self.state.transaction() as db:
                db.execute('UPDATE cs_jobs SET progress=? WHERE id=?', (canonical_json({'generated':len(outputs),'stage':'部分高阶产物被拒绝','errors':{k:str(v)[:120] for k,v in errors.items()}}), job['id']))
            unknown = bool(self.store.rows("SELECT 1 FROM cs_calls WHERE job_id=? AND state='unknown' LIMIT 1", (job['id'],)))
            raise ModelError('semantic_rejected' if 'experience_audit' in errors else 'derivation_rejected', '部分高阶记忆未通过结构化或来源校验，基础记忆完整保留', unknown=unknown)
        return {'generated':len(outputs), 'memory_ids':outputs, 'stage':'高阶记忆构建完成', 'omitted_window_sources':omitted}

    def derive_attachment_code(self,scope,sources,ing,job):
        from .repository import parse_source, RECIPE
        memories=[];edges=[];reused=0
        for source in sources:
            from .attachments import current_attachment
            if not current_attachment(self,source):continue
            name=source.metadata.get('attachment_name') or 'message.txt'
            code=source.text_content;line_offset=0
            if not source.metadata.get('attachment_id'):
                fence=re.search(r'```(python|py|typescript|ts|javascript|js)\s*\n(.*?)```',code,re.S)
                if fence:
                    name='message.'+{'python':'py','py':'py','typescript':'ts','ts':'ts','javascript':'js','js':'js'}[fence[1]];line_offset=code[:fence.start(2)].count('\n');code=fence[2]
                elif re.match(r'\s*(def |class |from |import )',code):name='message.py'
            parsed=parse_source(name,code)
            if parsed.get('error'):continue
            recipe=digest(['attachment-parser-v31',RECIPE,parsed['parser'],100,80,5000])
            for symbol in parsed['symbols'][:100]:
                start=symbol['start_line']+line_offset;end=min(symbol['end_line']+line_offset,start+79)
                snippet='\n'.join(source.text_content.splitlines()[start-1:end])[:5000]
                key='attachment_symbol:'+digest([source.logical_uid,source.version,symbol,recipe])
                existing=self.state.get_memories(scope,[key]).get(key)
                if existing:
                    if existing.metadata.get('recipe_fingerprint')!=recipe or existing.metadata.get('source_versions')!={source.logical_uid:source.version} or existing.raw_data.get('code')!=snippet:
                        raise AppError('源码派生身份冲突，未覆盖原始结果',409)
                    reused+=1;continue
                locator={'attachment_id':source.metadata.get('attachment_id'),'path':name,'start_line':start,'end_line':end,'file_hash':source.metadata.get('attachment_hash') or hashlib.sha256(source.text_content.encode()).hexdigest(),'repository_id':''}
                memories.append(CanonicalMemory(key,f"{symbol['kind']} {symbol['qualified_name']} · {name}:{start}-{end}\n{snippet}",raw_data={**symbol,'code':snippet,'parser':parsed['parser']},
                    metadata={'user_id':scope.user_id,'memory_family':'code','source_versions':{source.logical_uid:source.version},'source_locator':locator,'verification':'STATIC_DEFINITION','recipe_fingerprint':recipe},
                    session_id=ing['session'],request_id=source.request_id,created_at_ms=source.created_at_ms,role='derived',memory_type='code_symbol',memory_tower='engineering',memory_level='L1_Summary'))
                edges.append(Edge('attachment_support:'+digest([key,source.logical_uid]),key,source.logical_uid,'SUPPORTED_BY',provenance={'source_version':source.version}))
        if memories:
            self.coordinator.apply(scope,job['id']+':parse',digest([m.logical_uid for m in memories]),Mutation(memories=tuple(memories),edges=tuple(edges)),{})
            self.store.enqueue(job['owner'],ing['workspace'],'index',{'scopes':[asdict(scope)]},[job['id'],'attachment-index'])
        return {'generated':len(memories),'reused':reused,'memory_ids':[m.logical_uid for m in memories],'stage':'有界源码符号解析完成；未支持语言保留文本索引'}

    def visible(self,scope,memory):
        sessions={memory.session_id} if memory.session_id else set()
        refs=memory.metadata.get('source_versions',{}) or {r['uid']:r['version'] for r in memory.metadata.get('source_refs',[])}
        if refs:
            original=self.state.get_memories(Scope('chinasoft.raw.v1',scope.user_id),list(refs))
            sessions.update(m.session_id for m in original.values() if m.session_id)
        return not sessions or not self.store.rows('SELECT 1 FROM cs_sessions WHERE deleted=1 AND id IN ('+','.join('?' for _ in sessions)+')',list(sessions))

    def eligible(self, scope, memory):
        from .attachments import current_attachment
        if not current_attachment(self,memory):return False
        if not memory.active or memory.lifecycle=='deleted' or not self.visible(scope,memory): return False
        if memory.session_id and self.store.rows('SELECT 1 FROM cs_sessions WHERE id=? AND deleted=1',(memory.session_id,)):return False
        refs=memory.metadata.get('source_versions',{})
        if refs:
            raw=Scope('chinasoft.raw.v1',scope.user_id)
            original=self.state.get_memories(raw,list(refs))
            if any(not current_attachment(self,m) for m in original.values()):return False
            sessions={m.session_id for m in original.values() if m.session_id}
            if sessions and self.store.rows('SELECT 1 FROM cs_sessions WHERE deleted=1 AND id IN ('+','.join('?' for _ in sessions)+')',list(sessions)):return False
        if memory.commit_seq>self.state.scope_state(scope)['published_seq']:return False
        if not self.repositories.eligible(memory):return False
        if memory.metadata.get('hlm_reference'): return False
        if memory.metadata.get('hlm_kind'):
            raw = Scope('chinasoft.raw.v1', scope.user_id)
            reader=self.v3_hlm if memory.metadata.get('recipe_hash')==self.v3_recipe.fingerprint else self.hlm if memory.metadata.get('recipe_hash')==self.recipe.fingerprint else self.legacy_hlm
            return memory.logical_uid in reader.eligible(raw, {memory.logical_uid:memory})
        versions = memory.metadata.get('source_versions')
        if versions:
            raw = self.state.get_memories(scope, list(versions))
            def source_hash(source):
                if memory.memory_type in ('engineering_experience','engineering_knowledge'):
                    # ExperienceBuilder binds text AND origin metadata. A plain
                    # text SHA would reject its legitimate provenance contract.
                    from mandol_competition.code.software.episodes import source_digest
                    unit=MemoryUnit(uid=source.logical_uid,raw_data={'text_content':source.text_content},
                        metadata={'user_id':scope.user_id,'session_id':source.session_id,'request_id':source.request_id,
                                  'role':source.role,'ts_ms':source.business_timestamp_ms,'sequence':source.metadata.get('sequence')})
                    return source_digest(unit)
                return hashlib.sha256(source.text_content.encode()).hexdigest()
            if any(u not in raw or not raw[u].active or raw[u].version!=v or not self.repositories.eligible(raw[u]) or
                   (memory.metadata.get('source_hashes',{}).get(u) and source_hash(raw[u])!=memory.metadata['source_hashes'][u]) for u,v in versions.items()): return False
        if memory.memory_type=='visual_episode':
            from mandol_competition.multimodal.episodes import supported
            refs = memory.metadata.get('source_refs',[])
            return supported(memory,self.state.get_memories(scope,[r['uid'] for r in refs]))
        return True

    def serialize(self, scope, memory, *, detail=False):
        metadata = memory.metadata
        rows=self.store.rows('SELECT mode,workspace,repository FROM cs_scopes WHERE scope=?',(scope.key,))
        info=rows[0] if rows else {'mode':metadata.get('mode','textual'),'workspace':metadata.get('workspace_id',''),'repository':metadata.get('repository_id','')}
        mode=MODES.get(metadata.get('memory_family'),info['mode']) if not is_raw(memory) else info['mode']
        level = 'L0' if memory.memory_level=='L0_Observation' else 'L2' if memory.memory_level.startswith('L2') or memory.memory_type=='profile' else 'L1'
        result = {'id':memory.logical_uid, 'text':memory.text_content if detail else memory.text_content[:650],
                  'level':level, 'actual_level':memory.memory_level, 'type':memory.memory_type, 'tower':memory.memory_tower,
                  'session_id':memory.session_id,'version':memory.version,'created_at':memory.created_at_ms,
                  'role':memory.role,'attachment_id':metadata.get('attachment_id'),'attachment_name':metadata.get('attachment_name'),'attachment_kind':metadata.get('attachment_kind'),
                  'media_ids':metadata.get('media_ids',[]), 'source_versions':metadata.get('source_versions',{}),
                  'eligible':self.eligible(scope,memory), 'verification':metadata.get('verification','SOURCE_CITED' if level!='L0' else 'ORIGINAL'),
                  'generation':metadata.get('generation'), 'lexical_ready':bool(memory.active and memory.commit_seq<=self.state.scope_state(scope)['published_seq'])}
        if not result['session_id'] and metadata.get('source_versions'):
            originals=self.state.get_memories(Scope('chinasoft.raw.v1',scope.user_id),list(metadata['source_versions']))
            sessions={m.session_id for m in originals.values() if m.session_id}
            if len(sessions)==1:result['session_id']=next(iter(sessions))
        result.update(mode=mode,workspace_id=info['workspace'],memory_family=FAMILIES[mode],memory_type=view_type(memory,mode),
                      source_kind='raw' if is_raw(memory) else 'derived',repository_id=metadata.get('repository_id') or info['repository'],
                      lifecycle='active' if result['eligible'] else 'invalid',recipe_fingerprint=metadata.get('recipe_fingerprint',metadata.get('recipe_hash')),
                      source_locator=metadata.get('source_locator'))
        if mode=='textual' and memory.memory_type=='profile':
            result['profile_status']=metadata.get('hlm_item',{}).get('profile_item',{}).get('status')
        if metadata.get('code_associations'):
            associations=[]
            for binding in metadata['code_associations']:
                current=self.state.get_memories(scope,[binding['memory_id']]).get(binding['memory_id'])
                associations.append(dict(binding,current_reference_valid=bool(current and current.active and self.repositories.eligible(current))))
            result['code_associations']=associations
        state = self.state.scope_state(scope)
        rows = self.state._rows('SELECT published_seq AS seq,status FROM projection_frontiers WHERE scope=? AND channel=?', (scope.key,self.projection_channel))
        result['dense_ready'] = bool(rows and rows[0]['seq']==state['published_seq'] and rows[0]['status']=='ready')
        from .memory_quality import validity
        result['quality']=validity(self,scope,memory,eligible=result['eligible'])
        if detail:
            result.update(raw_data=memory.raw_data, metadata=metadata)
            result['sources'] = self.sources(scope,memory)
        return result

    def sources(self, scope, memory):
        refs = memory.metadata.get('source_versions', {})
        evidence = memory.metadata.get('hlm_item', {}).get('evidence', memory.metadata.get('evidence',memory.raw_data.get('evidence',[])))
        if not refs: refs = {r['uid']:r['version'] for r in memory.metadata.get('source_refs',[])}
        raw_scope = Scope('chinasoft.raw.v1', scope.user_id)
        originals = self.state.get_memories(raw_scope,list(refs)); result=[]
        for key, version in list(refs.items())[:64]:
            m=originals.get(key)
            if m and self.visible(raw_scope,m):
                result.append({'id':key,'version':m.version,'expected_version':version,'text':m.text_content,
                               'session_id':m.session_id,'role':m.role,'media_ids':m.metadata.get('media_ids',[]),
                               'quotes':[e['quote'] for e in evidence if e.get('source_uid')==key], 'active':self.eligible(raw_scope,m),
                               'source_locator':m.metadata.get('source_locator'),'memory_type':m.memory_type,
                               'repository_id':m.metadata.get('repository_id','')})
        return result

    def locate(self, owner, workspace, key,mode=None,repository_id=None):
        for scope, info in self.store.scopes(owner,workspace,mode,repository_id):
            memories=self.state.get_memories(scope,[key])
            if key in memories and self.visible(scope,memories[key]) and memories[key].active and memories[key].commit_seq<=self.state.scope_state(scope)['published_seq'] and not memories[key].metadata.get('hlm_reference'): return scope,memories[key],info
        raise AppError('记忆不存在或无权访问',404,code='not_found')

    def search(self, owner, workspace, query, mode=None, limit=8, *, repository_id=None,retrieval_scope='current',memory_type=None,source_kind=None,session_id=None,conversation_first=False,exclude_sources=None,strategy=None):
        selected=strategy or self.settings.memory_query_strategy
        kwargs=dict(repository_id=repository_id,retrieval_scope=retrieval_scope,memory_type=memory_type,source_kind=source_kind,
                    session_id=session_id,conversation_first=conversation_first,exclude_sources=exclude_sources)
        observed=[];trace_var=getattr(self.retrieval,'request_trace',None);token=trace_var.set(observed) if trace_var else None
        try:
            if selected in ('baseline','B0_current'):result=self._search_baseline(owner,workspace,query,mode,limit,**kwargs)
            else:
                from .typed_retrieval import TypedRetrieval
                from .query_planner import POLICIES
                if selected not in POLICIES:raise AppError('未知查询策略')
                result=TypedRetrieval(self).search(owner,workspace,query,mode,limit,strategy=selected,**kwargs)
            if trace_var:
                trace=result.setdefault('route',{}).setdefault('trace',{})
                trace['model_http']=observed[:24];trace['model_http_requests']=len(observed)
            return result
        finally:
            if trace_var:trace_var.reset(token)

    def _search_baseline(self, owner, workspace, query, mode=None, limit=8, *, repository_id=None,retrieval_scope='current',memory_type=None,source_kind=None,session_id=None,conversation_first=False,exclude_sources=None):
        started=time.monotonic();timings={};consumed={k:0 for k in ('bm25','cosine','splade')}
        excluded=set(exclude_sources or [])
        if source_kind not in (None,'','raw','derived'):raise AppError('无效来源分类')
        mode=MODES.get(mode,mode or 'textual');plan=plan_query(workspace,mode,repository_id,retrieval_scope,query)
        if repository_id:self.store.repository(owner,workspace,repository_id)
        scopes=self.store.scopes(owner,workspace,None if retrieval_scope=='workspace' or conversation_first else mode,None if retrieval_scope=='workspace' or conversation_first else repository_id or '')
        if conversation_first:
            from .visualization import MemoryViews
            if session_id:self.store.session(owner,workspace,session_id)
            views=MemoryViews(self)
            scopes=[(s,i) for s,i in scopes if not i['repository'] or i['repository']==repository_id]
        candidates={}; multi=MultiRetrievalResults(); warnings=[]
        if not scopes or not self.state._rows('SELECT 1 FROM memories WHERE active=1 AND scope IN ('+','.join('?' for _ in scopes)+') LIMIT 1', [s.key for s,_ in scopes]):
            return {'items':[],'warnings':[],'query':query,'route':plan.public()}
        try:
            vector,sparse=self.retrieval.encode_query(query)
            timings.update(getattr(self.retrieval.local,'timings',{}))
        except ModelError: vector,sparse=None,None;warnings.append('Dense / Sparse 查询不可用，使用真实 BM25 结果')
        query_started=time.monotonic()
        groups={method:[] for method in ('bm25','cosine','splade')};exact_ids=set()
        for scope, info in scopes:
            self.state.ensure_scope(scope)
            with self.coordinator.search_view(scope) as view:
                allowed=None
                if memory_type or source_kind or conversation_first or excluded:
                    allowed=[]
                    mapping=TYPE_MAP.get(info['mode'],{});physical=[k for k,v in mapping.items() if v==memory_type]
                    predicate="scope=? AND active=1";params=[scope.key]
                    if excluded:
                        predicate+=' AND logical_uid NOT IN ('+','.join('?' for _ in excluded)+')';params.extend(sorted(excluded))
                    if memory_type:
                        if not physical:continue
                        predicate+=' AND memory_type IN ('+','.join('?' for _ in physical)+')';params.extend(physical)
                    if source_kind:predicate+=" AND memory_level"+("='L0_Observation'" if source_kind=='raw' else "<>'L0_Observation'")
                    if conversation_first and session_id and not info['repository']:
                        predicate+=" AND (session_id=? OR EXISTS (SELECT 1 FROM json_each(memories.metadata,'$.source_versions') ref JOIN memories src ON src.logical_uid=ref.key JOIN cs_scopes own ON own.scope=src.scope WHERE src.session_id=? AND own.owner=? AND own.workspace=?))"
                        params.extend([session_id,session_id,owner,workspace])
                    allowed=[r['logical_uid'] for r in self.state._rows('SELECT logical_uid FROM memories WHERE '+predicate,params)]
                    if excluded:
                        checked=self.state.get_memories(scope,allowed)
                        allowed=[key for key,m in checked.items() if not (set(m.metadata.get('source_versions',{}))|{r['uid'] for r in m.metadata.get('source_refs',[])})&excluded]
                    if conversation_first:
                        check=self.state.get_memories(scope,allowed)
                        allowed=[key for key,m in check.items() if (not session_id or info['repository'] or views.belongs(scope,m,session_id)) and (not m.session_id or not self.store.rows('SELECT 1 FROM cs_sessions WHERE id=? AND deleted=1',(m.session_id,)))]
                hits=self.state.lexical.search(scope,query,limit=24,published_seq=view.published_seq,candidate_uids=allowed)
                groups['bm25'].extend((scope,info,h) for h in hits)
                if info['mode']=='code' and info['repository']:
                    from types import SimpleNamespace
                    exact=self.repositories.exact_candidates(info['repository'],query)
                    exact=[key for key in exact if allowed is None or key in allowed];exact_ids.update(exact)
                    groups['bm25'].extend((scope,info,SimpleNamespace(logical_uid=key,score=100.0)) for key in exact)
                frontier=self.state._rows('SELECT published_seq AS seq,status FROM projection_frontiers WHERE scope=? AND channel=?',(scope.key,self.projection_channel))
                if vector is not None and frontier and frontier[0]['seq']==view.published_seq and frontier[0]['status']=='ready':
                    try:
                        for method, value in (('cosine',vector),('splade',sparse)):
                            groups[method].extend((scope,info,h) for h in self.projection.search(scope,method,value,limit=24,published_seq=view.published_seq,candidate_uids=allowed))
                    except Exception: warnings.append('部分向量投影不可用，保留基础检索结果')
        timings['candidate_search_ms']=round((time.monotonic()-query_started)*1000,2)
        methods={'bm25':RetrievalMethod.BM25,'cosine':RetrievalMethod.COSINE_SIMILARITY,'splade':RetrievalMethod.SPLADE}
        for method, rows in groups.items():
            for scope,info,hit in sorted(rows,key=lambda x:(-x[2].score,x[2].logical_uid))[:40]:
                memory=self.state.get_memories(scope,[hit.logical_uid]).get(hit.logical_uid)
                if not memory or not self.eligible(scope,memory):continue
                candidates[memory.logical_uid]=(scope,memory,info)
                unit=MemoryUnit(uid=memory.logical_uid,raw_data={'text_content':memory.text_content})
                multi.add_result(RetrievalResult(unit,hit.score,methods[method]))
                consumed[method]+=1
        fused=ScoreFusion.rrf_fusion(multi,k=60,top_k=16)
        fused_count=len(fused);reranker_used=False;rank_started=time.monotonic()
        if fused:
            try:
                ranked=self.retrieval.rerank(query,[candidates[u.uid][1].text_content[:5000] for u,_ in fused]);fused=[(fused[i][0],score) for i,score in ranked];reranker_used=True
            except ModelError:warnings.append('BAAI 重排不可用，采用 RRF 排序')
        timings['rerank_ms']=round((time.monotonic()-rank_started)*1000,2)
        # Preserve exact code matches and use scenario-specific priorities within a bounded candidate set.
        fused=sorted(fused,key=lambda pair:(0 if pair[0].uid in exact_ids else 1,0 if view_type(candidates[pair[0].uid][1],candidates[pair[0].uid][2]['mode']) in plan.preferred_types else 1,-pair[1]))
        items=[]
        for unit,score in fused[:limit]:
            scope,memory,info=candidates[unit.uid]
            current = self.state.get_memories(scope,[memory.logical_uid]).get(memory.logical_uid)
            if not current or current.version != memory.version or current.text_content != memory.text_content: continue
            # Recheck source versions after remote ranking; stale derived facts are excluded.
            if not self.eligible(scope,memory):continue
            item=self.serialize(scope,memory,detail=True);item.update(score=score);items.append(item)
        timings['total_ms']=round((time.monotonic()-started)*1000,2)
        trace={'profile':self.vector_profile.hash,'collection':self.vector_collection,'sparse_model':self.vector_profile.splade_model,
               'candidate_counts':{k:len(v) for k,v in groups.items()},'rrf_consumed':consumed,'rrf_k':60,'rrf_candidates':fused_count,
               'reranker':self.settings.reranker_model,'reranker_used':reranker_used,'timings':timings,
               'three_channels_consumed':all(consumed.values()) and self.retrieval.probe_sparse()['ready']}
        if self.settings.retrieval_backend=='vllm':
            from .retrieval_contract import MODELS,recipe_hash
            trace.update(serving='native-vllm-gpu-http',models={c:{'model':m['model'],'revision':m['revision'],'recipe_hash':recipe_hash(c)} for c,m in MODELS.items()},
                         retrieval_complete=trace['three_channels_consumed'] and reranker_used)
        for key,value in timings.items():
            if key.endswith('_ms'):self.store.metric('retrieval_'+key,value)
        return {'items':items,'warnings':sorted(set(warnings)), 'query':query,'route':{**plan.public(),'conversation_id':session_id,'conversation_first':conversation_first,'searched_scopes':len(scopes),'searched_families':sorted({FAMILIES[i['mode']] for _,i in scopes}),'searched_repositories':sorted({i['repository'] for _,i in scopes if i['repository']}),'trace':trace}}

    def explorer(self, owner, workspace, mode=None, level=None, session=None, keyword='', offset=0, limit=24,*,repository_id=None,memory_type=None,source_kind=None,retrieval_scope='current',state_view=None):
        mode=MODES.get(mode,mode)
        if repository_id:self.store.repository(owner,workspace,repository_id)
        if retrieval_scope not in ('current','workspace'):raise AppError('无效检索范围')
        # Internal legacy callers retain all-family inspection. Public routes always supply a family.
        scopes=self.store.scopes(owner,workspace,None if retrieval_scope=='workspace' else mode,None if retrieval_scope=='workspace' or mode is None else repository_id or '');scope_map={s.key:(s,info) for s,info in scopes}
        if not scope_map:return {'items':[],'total':0,'offset':offset,'limit':limit}
        where=['m.scope IN ('+','.join('?' for _ in scope_map)+')','m.active=1','m.commit_seq<=s.published_seq',"m.memory_type<>'reference'"]
        params=list(scope_map)
        if session:where.append('m.session_id=?');params.append(session)
        if memory_type:
            physical=list({k for mapping in TYPE_MAP.values() for k,v in mapping.items() if v==memory_type})
            if not physical:return {'items':[],'total':0,'offset':offset,'limit':limit}
            where.append('m.memory_type IN ('+','.join('?' for _ in physical)+')');params.extend(physical)
        if source_kind:
            if source_kind not in ('raw','derived'):raise AppError('无效来源分类')
            where.append("m.memory_level"+("='L0_Observation'" if source_kind=='raw' else "<>'L0_Observation'"))
        if keyword:where.append('m.text_content LIKE ? ESCAPE \'\\\'');params.append('%'+keyword.replace('\\','\\\\').replace('%','\\%').replace('_','\\_')+'%')
        if level=='L0':where.append("m.memory_level='L0_Observation'")
        elif level=='L2':where.append("(m.memory_level LIKE 'L2%' OR m.memory_type='profile')")
        elif level=='L1':where.append("m.memory_level<>'L0_Observation' AND m.memory_level NOT LIKE 'L2%' AND m.memory_type<>'profile'")
        if state_view is not None and state_view not in ('current','history','all'):raise AppError('无效记忆状态视图')
        if state_view in ('history','all'):where.remove('m.active=1')
        sql=' FROM memories m JOIN scopes s USING(scope) WHERE '+' AND '.join(where)
        total=self.state._rows('SELECT COUNT(*) n'+sql,params)[0]['n'];candidate_total=total
        rows=self.state._rows('SELECT m.scope,m.logical_uid'+sql+' ORDER BY m.created_at_ms DESC,m.logical_uid LIMIT ? OFFSET ?',params+([5001,0] if state_view is not None else [limit,offset]))
        items=[]
        for row in rows:
            scope,info=scope_map[row['scope']];m=self.state.get_memories(scope,[row['logical_uid']])[row['logical_uid']]
            if m.metadata.get('hlm_reference') or not self.visible(scope,m):continue
            if state_view=='current' and not self.eligible(scope,m):continue
            if state_view=='history' and self.eligible(scope,m):continue
            item=self.serialize(scope,m);item['mode']=info['mode'];items.append(item)
        if state_view is not None:
            total=len(items);items=items[offset:offset+limit]
        return {'items':items,'total':total,'offset':offset,'limit':limit,**({'selection':{'complete':candidate_total<=5000,'candidates':candidate_total,'scanned':min(candidate_total,5000)}} if state_view is not None else {})}

    def save_media(self, owner, workspace, session, data):
        with self.media_lock:return self._save_media(owner,workspace,session,data)

    def _save_media(self, owner, workspace, session, data):
        selected=self.store.session(owner,workspace,session)
        if not selected['unified'] and selected['mode']!='multimodal':raise AppError('旧会话需要视觉模式或新建统一对话')
        if len(data)>self.settings.max_media_bytes:raise AppError('图片最大 8 MiB',413)
        try:
            with Image.open(io.BytesIO(data)) as im:
                if im.format not in ('PNG','JPEG','WEBP') or im.width*im.height>20000000:raise AppError('只支持 PNG/JPEG/WebP，最多 2000 万像素',415)
                im.verify()
            image=Image.open(io.BytesIO(data));image.load();mime={'PNG':'image/png','JPEG':'image/jpeg','WEBP':'image/webp'}[image.format]
        except (UnidentifiedImageError,OSError,Image.DecompressionBombError) as exc:raise AppError('图片文件损坏或类型不支持',415) from exc
        sha=hashlib.sha256(data).hexdigest();mid='img_'+digest([owner,workspace,session,sha])
        old=self.store.rows('SELECT id FROM cs_media WHERE id=?',(mid,))
        if old:return self.store.media(owner,mid)
        with self.state.transaction() as db:self.store.access.check(db,owner,'storage_bytes',len(data))
        relative=Path('media')/owner/(mid+'.bin');thumb=relative.with_suffix('.jpg');path=self.settings.data_dir/relative
        path.parent.mkdir(parents=True,exist_ok=True)
        for directory in (path.parent.parent,self.settings.data_dir):
            descriptor=os.open(directory,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(descriptor)
            finally:os.close(descriptor)
        def durable_write(p, content):
            temporary=p.parent/('.media-'+uid('')+'.tmp')
            try:
                with temporary.open('xb') as f:f.write(content);f.flush();os.fsync(f.fileno())
                os.replace(temporary,p)
            finally: temporary.unlink(missing_ok=True)
        durable_write(path,data)
        preview=image.convert('RGB');preview.thumbnail((480,480));buf=io.BytesIO();preview.save(buf,format='JPEG',quality=82);durable_write(self.settings.data_dir/thumb,buf.getvalue())
        descriptor=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)
        with self.state.transaction() as db:
            allowed=db.execute('SELECT 1 FROM cs_sessions s JOIN cs_workspaces w ON w.id=s.workspace WHERE s.id=? AND s.owner=? AND w.id=? AND w.demo<>-1',(session,owner,workspace)).fetchone()
            if not allowed:
                path.unlink(missing_ok=True);(self.settings.data_dir/thumb).unlink(missing_ok=True)
                raise AppError('工作区已删除或正在重置，图片未被接收',409)
            try:
                self.store.access.check(db,owner,'storage_bytes',len(data))
                self.store.access.check(db,owner,'media_files')
            except AppError:
                path.unlink(missing_ok=True);(self.settings.data_dir/thumb).unlink(missing_ok=True)
                raise
            db.execute('INSERT OR IGNORE INTO cs_media VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',(mid,owner,workspace,session,sha,str(relative),str(thumb),mime,image.width,image.height,len(data),now()))
        return self.store.media(owner,mid)

    def status(self, owner, workspace):
        self.store.workspace(owner,workspace)
        qdrant=True
        try:self.projection.health()
        except Exception:qdrant=False
        counts=self.store.rows('SELECT status,COUNT(*) n FROM cs_jobs WHERE owner=? AND workspace=? GROUP BY status',(owner,workspace))
        metrics=self.store.rows('SELECT name,COUNT(*) samples,ROUND(AVG(milliseconds),2) avg_ms,ROUND(MAX(milliseconds),2) max_ms FROM cs_metrics GROUP BY name')
        workers_alive = all(t.is_alive() for t in self.threads)
        model_health=self.retrieval.probe_all(force=True) if self.settings.retrieval_backend=='vllm' else {'sparse':self.retrieval.probe_sparse(force=True)}
        sparse=model_health['sparse']
        all_scopes=self.store.scopes(owner,workspace)
        scopes=[(scope,info) for scope,info in all_scopes if self.state._rows("SELECT 1 FROM memories WHERE scope=? AND lifecycle<>'deleted' LIMIT 1",(scope.key,))];ready=0
        if scopes:
            try:self.projection.ensure_schema(scopes[0][0],create=False)
            except Exception:qdrant=False
        for scope,_ in scopes:
            frontier=self.state._rows('SELECT published_seq,status FROM projection_frontiers WHERE scope=? AND channel=?',(scope.key,self.projection_channel))
            if frontier and frontier[0]['status']=='ready' and frontier[0]['published_seq']==self.state.scope_state(scope)['published_seq']:ready+=1
        retrieval={'profile':self.vector_profile.hash,'collection':self.vector_collection,'sparse':sparse,
                   'ready_scopes':ready,'total_scopes':len(scopes),'empty_scopes':len(all_scopes)-len(scopes),'three_channel_ready':all(h['ready'] for h in model_health.values()) and qdrant and ready==len(scopes),
                   'models':model_health,'serving':self.settings.retrieval_backend,
                   'allow_degraded':self.settings.allow_retrieval_degraded}
        return {'status':'healthy' if qdrant and workers_alive and retrieval['three_channel_ready'] else 'degraded','sqlite':self.state.health(),'qdrant':qdrant,'retrieval':retrieval,
                'config':self.settings.public(),'jobs':{r['status']:r['n'] for r in counts},'metrics':metrics,
                'workers':[{'name':t.name,'alive':t.is_alive()} for t in self.threads], 'uptime_ms':now()-self.started}
