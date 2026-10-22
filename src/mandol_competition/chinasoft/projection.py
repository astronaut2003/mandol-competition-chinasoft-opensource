"""Versioned optional retrieval projection over the existing canonical owner.

This adapter does not migrate canonical profiles, namespaces, recipes, or facts.
It reuses the substrate Qdrant codec/reconciliation in a separate collection and
frontier. The read-through state view owns only projection manifests/frontiers;
semantic writes are prohibited. Old collections remain available for rollback.
"""
from dataclasses import replace, asdict
import json, time, sqlite3
from mandol_substrate.contracts import ResourceIdentity, PublicationPolicy
from mandol_substrate.publication import PublicationCoordinator
from mandol_substrate.qdrant.backend import QdrantProjectionBackend
from mandol_substrate.qdrant.policy import QdrantProjectionPolicy
from mandol_substrate.vector import MinimalProjectionCodec
from mandol_substrate.state import RetrievalProfile, Scope
from mandol_substrate.state.models import canonical_json, StateUnavailable, digest
from .models import RetrievalModels
from .store import now


def canonical_profile(settings, proposed):
    """Open the persisted semantic contract, independently of vector upgrades."""
    file=settings.data_dir/'state.sqlite3'
    if not file.exists():return proposed
    with sqlite3.connect('file:'+str(file)+'?mode=ro',uri=True) as db:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='scopes'").fetchone():return proposed
        row=db.execute('SELECT profile_json FROM scopes LIMIT 1').fetchone()
    if not row:return proposed
    data=json.loads(row[0]);data['methods']=tuple(data.get('methods',('cosine','bm25','splade')))
    existing=RetrievalProfile(**data)
    # Canonical history/derivation recipes retain their original identity. A
    # native vLLM upgrade changes only the independently versioned projection.
    if settings.retrieval_backend == 'vllm':
        if existing.tokenizer_fingerprint != proposed.tokenizer_fingerprint:
            raise ValueError('Canonical lexical tokenizer differs; explicit migration required')
        return existing
    if (existing.embedding_model,existing.embedding_dimension,existing.tokenizer_fingerprint)!=(proposed.embedding_model,proposed.embedding_dimension,proposed.tokenizer_fingerprint):
        raise ValueError('Canonical model/tokenizer differs; an explicit semantic migration is required')
    return existing


class ProjectionState:
    def __init__(self,canonical,profile,collection):
        self.canonical=canonical;self.profile_hash=profile.hash
        self.identity=ResourceIdentity('chinasoft','chinasoft.raw.v1',collection)
        self.prefix='chinasoft.projection.v31:'+profile.hash+':'
        self.channel='retrieval.'+profile.hash

    def __getattr__(self,name):
        if name in ('prepare','publish','ensure_scope','authorize_derived_namespace'):
            raise RuntimeError('Projection state cannot mutate semantic state')
        return getattr(self.canonical,name)

    def bind_manifest(self,key,value):return self.canonical.bind_manifest(self.prefix+key,value)

    def _rows(self,sql,params=()):
        # The backend's one bootstrap guard is namespaced to this projection.
        if sql=="SELECT 1 FROM manifest WHERE key='projection_collection_initialized'":
            return self.canonical._rows('SELECT 1 FROM manifest WHERE key=?',(self.prefix+'projection_collection_initialized',))
        return self.canonical._rows(sql,params)

    def set_projection_frontier(self,scope,channel,seq):return self.canonical.set_projection_frontier(scope,self.channel,seq)
    def mark_projection_degraded(self,scope,channel,error):return self.canonical.mark_projection_degraded(scope,self.channel,error)


class VersionedProjection:
    def __init__(self,platform,deferred):
        p=platform;s=p.settings
        self.profile=replace(p.profile,splade_model=RetrievalModels.SPLADE_MODEL,
            splade_revision=RetrievalModels.SPLADE_REVISION,
            options={'deployment':'chinasoft-retrieval-v31','sparse_endpoint':s.sparse_url,
                     'embedding_endpoint':s.embedding_url,'reranker':s.reranker_model,
                     'pooling':RetrievalModels.SPLADE_POOLING,'query_mode':'full','document_mode':'full',
                     'weights_sha256':RetrievalModels.SPLADE_WEIGHTS,'tokenizer_sha256':RetrievalModels.SPLADE_TOKENIZER,
                     'config_sha256':RetrievalModels.SPLADE_CONFIG,
                     'vocabulary_size':30522,'max_tokens':512,'input_char_prefix':4000,
                     'document_and_query_encoding':'separate-explicit-methods-v31'})
        self.collection=s.collection+'_splade_'+self.profile.hash[:12]
        if s.retrieval_backend == 'vllm':
            from .retrieval_contract import MODELS, recipe, recipe_hash
            dense=MODELS['dense'];sparse=MODELS['sparse']
            # Endpoint/GPU/concurrency placement does not change model math.
            # BF16 SPLADE and Qwen replace the old FP32/BGE numerical contract,
            # so never reuse the old CPU/cloud projection despite equal widths.
            self.profile=replace(p.profile,embedding_model=dense['model'],embedding_revision=dense['revision'],
                embedding_dimension=1024,splade_model=sparse['model'],splade_revision=sparse['revision'],
                options={'deployment':'chinasoft-native-vllm-gpu-v1',
                         'recipes':{c:recipe(c) for c in MODELS},'recipe_hashes':{c:recipe_hash(c) for c in MODELS},
                         'input_char_prefix':4000,'document_and_query_encoding':'separate-explicit-methods-v31'})
            self.collection=s.collection+'_vllm_'+self.profile.hash[:12]
        self.state=ProjectionState(p.state,self.profile,self.collection)
        self.channel=self.state.channel
        policy=QdrantProjectionPolicy()
        self.backend=deferred(lambda:QdrantProjectionBackend(s.qdrant_url,self.profile,state=self.state,
            storage_id=p.state.db_uuid,identity=self.state.identity,collection_name=self.collection,optional_projection=True))
        self.coordinator=PublicationCoordinator(self.state,self.backend,MinimalProjectionCodec(self.profile,policy.hash),p.retrieval,
            gates=p.coordinator.gates,policy=PublicationPolicy('chinasoft-versioned-optional-retrieval-v31',required_projection=False,encode_before_prepare=False),
            retries=2,verify_timeout=25,batch_size=8)
        self.platform=p
        self.last_schedule=0

    def project(self,scope,job=None):
        p=self.platform;c=self.coordinator;started=time.monotonic()
        try:
            with p.coordinator.gates.lease(scope,write=True):
                current=p.state.scope_state(scope);seq=current['published_seq']
                if seq!=current['canonical_seq'] or seq!=current['lexical_seq']:return False
            self.backend.ensure_schema(scope,create=True)
            indexed=0;encoded=0
            for batch in p.state.iter_memories(scope,8):
                if p.stop_event.wait(.02):return False
                actual=self.backend.get_projection_metadata(scope,[m.logical_uid for m in batch])
                missing=[m for m in batch if not c.codec.matches(actual.get(m.logical_uid),c.codec.metadata(scope,m))]
                prepared={}
                if missing:
                    # Encoding outside scope gates and all SQLite transactions.
                    vectors=p.retrieval.encode_documents([m.text_content for m in missing]);encoded+=len(missing)
                    prepared={m.logical_uid:v for m,v in zip(missing,vectors,strict=True)}
                with p.coordinator.gates.lease(scope,write=True):
                    if p.state.scope_state(scope)['published_seq']!=seq:
                        raise StateUnavailable('canonical source changed during projection')
                    c._reconcile_batch(scope,seq,[{'logical_uid':m.logical_uid,'operation':'upsert'} for m in batch],prepared)
                indexed+=len(batch)
                if job:
                    with p.state.transaction() as db:
                        db.execute('UPDATE cs_jobs SET progress=?,updated_at=? WHERE id=? AND lease=?',
                            (canonical_json({'stage':'SPLADE 版本化投影补建','indexed':indexed,'encoded':encoded,'snapshot_seq':seq,'profile':self.profile.hash}),now(),job['id'],job['lease']))
            with p.coordinator.gates.lease(scope,write=True):
                if p.state.scope_state(scope)['published_seq']!=seq:raise StateUnavailable('source changed')
                self.backend.audit_rows(scope)
                self.backend.gc_scope(scope)
                p.state.set_projection_frontier(scope,self.channel,seq)
            p.store.metric('projection_v31',(time.monotonic()-started)*1000)
            return True
        except Exception as exc:
            p.state.mark_projection_degraded(scope,self.channel,type(exc).__name__)
            return False

    def schedule(self,limit=6):
        """Bounded durable backfill; failed jobs require explicit retry, never a spin."""
        if time.monotonic()-self.last_schedule<2:return 0
        self.last_schedule=time.monotonic()
        p=self.platform;added=0
        rows=p.store.rows("SELECT c.* FROM cs_scopes c JOIN scopes s ON s.scope=c.scope JOIN cs_workspaces w ON w.id=c.workspace AND w.owner=c.owner AND w.demo<>-1 WHERE s.published_seq>0 AND EXISTS (SELECT 1 FROM memories m WHERE m.scope=c.scope AND m.lifecycle<>'deleted') AND NOT EXISTS (SELECT 1 FROM projection_frontiers f WHERE f.scope=c.scope AND f.channel=? AND f.status='ready' AND f.published_seq=s.published_seq) ORDER BY c.scope",(self.channel,))
        for row in rows:
            scope=Scope(row['namespace'],row['user_id'])
            identity=['retrieval-projection-v31',self.profile.hash,scope.key,p.state.scope_state(scope)['published_seq']]
            key='job_'+digest([row['owner'],row['workspace'],'index',identity])
            if p.store.rows('SELECT 1 FROM cs_jobs WHERE id=?',(key,)):continue
            try:
                p.store.enqueue(row['owner'],row['workspace'],'index',{'scopes':[asdict(scope)],'projection_profile':self.profile.hash},
                                identity)
                added+=1
            except Exception:break
            if added>=limit:break
        return added
