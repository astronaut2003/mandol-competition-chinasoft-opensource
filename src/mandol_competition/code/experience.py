"""Opt-in, query-independent experience formation over the existing substrate.

No new database, no Search generation, no outcome inferred as verified. Raw
Code acknowledgement remains durable canonical+lexical, even if derivation or
optional Dense fails. Completed and unknown provider outcomes are never blindly
repeated on replay. Exact citations establish lineage, not semantic entailment.
"""
import json
import os

from mandol_substrate.representation import MemoryUnit
from mandol_substrate.state import BuildJob, CanonicalMemory, Edge, Mutation
from mandol_substrate.state.ledger import RequestLedger
from mandol_substrate.state.models import digest
from .software.episode_distillation import EngineeringEpisodeDistiller, INSTRUCTION
from .software.episodes import EngineeringEpisodeBuilder, EngineeringEpisodeMemory

RECIPE = 'code-source-cited-experience-v1'
FINGERPRINT = digest({'recipe': RECIPE, 'prompt': INSTRUCTION, 'max_input_chars': 32000,
                      'selection': 'multi-turn engineering session with >=256 source chars',
                      'generator': 'gpt-4o-mini-closeai', 'schema': 'engineering-episode-v2'})


def load_candidate():
    path = os.environ.get('MANDOL_CODE_QUALITY_CANDIDATE')
    if not path:
        return None
    from pathlib import Path
    value = json.loads(Path(path).read_text())
    expected = {'schema','id','recipe','distill','route'}
    if value.get('schema') == 'mandol-code-distinctive-anchor-candidate-v1':
        if set(value)!=expected|{'anchor_policy'} or value.get('anchor_policy')!='episode-anchor-idf-v1':
            raise ValueError('Invalid distinctive Code anchor policy')
        if value.get('distill') is not False or value.get('route')!='raw':
            raise ValueError('Distinctive anchor control must retain the raw lane')
        expected |= {'anchor_policy'}
        schema = 'mandol-code-distinctive-anchor-candidate-v1'
    else:
        schema = 'mandol-code-experience-candidate-v1'
    if (set(value) != expected
            or value['schema'] != schema
            or not isinstance(value['id'], str) or not value['id']
            or value['recipe'] != RECIPE or type(value['distill']) is not bool
            or value['route'] not in ('raw','gated_hybrid')):
        raise ValueError('Invalid explicitly versioned Code experience candidate')
    return value


def source_units(rows, user):
    result = []
    for row in rows:
        payload = json.loads(row['payload'])
        result.append(MemoryUnit(uid=row['uid'], raw_data={'text_content': payload['content']},
            metadata={'user_id': user, 'session_id': row['session_id'], 'request_id': row['request_id'],
                      'role': payload['role'], 'ts_ms': payload.get('timestamp'), 'sequence': row['sequence']}))
    return result


class ExperienceBuilder:
    def __init__(self, service, *, generator=None, model_plan=None):
        self.model_plan = model_plan
        self.fingerprint = FINGERPRINT if model_plan is None else digest({
            'recipe': RECIPE, 'prompt': INSTRUCTION, 'max_input_chars': 32000,
            'selection': 'multi-turn engineering session with >=256 source chars',
            'model_plan': model_plan, 'schema': 'engineering-episode-v2'})
        self.service, self.store = service, service.store
        self.ledger = RequestLedger(self.store.state)
        self.distiller = EngineeringEpisodeDistiller()
        self.http = None
        self.generate = generator or self._generate

    def _generate(self, prepared, request_id):
        if self.http is None:
            from mandol_competition.inference.vlm import HTTPVLMClient
            plan = self.model_plan or {}
            endpoint = plan.get('base_url') or os.environ.get('CHINASOFT_LLM_BASE_URL' if self.model_plan else 'MANDOL_CODE_LLM_BASE_URL')
            key_env = plan.get('api_key_env') or os.environ.get('CHINASOFT_LLM_API_KEY_ENV' if self.model_plan else 'MANDOL_CODE_LLM_API_KEY_ENV', 'DEEPSEEK_API_KEY' if self.model_plan else 'CLOSEAI_API_KEY')
            key = os.environ.get(key_env)
            wire = plan.get('request_model') or os.environ.get('CHINASOFT_LLM_REQUEST_MODEL' if self.model_plan else 'MANDOL_CODE_LLM_REQUEST_MODEL', 'deepseek-flash' if self.model_plan else 'gpt-4o-mini')
            if not endpoint or not key or not wire:
                raise ValueError('Explicit endpoint/credential/model mapping required')
            self.http = HTTPVLMClient(endpoint, wire, api_key=key, timeout=90)
        result = self.http.complete([
            {'role':'system','content':prepared['instruction']},
            {'role':'user','content':json.dumps({k:v for k,v in prepared.items() if k!='instruction'}, ensure_ascii=False)}],
            request_id=request_id, response_format={'type':'json_object'}, temperature=0, max_tokens=1800)
        if result.get('model') not in ((self.model_plan or {}).get('accepted_response_models') or ('gpt-4o-mini','gpt-4o-mini-closeai','gpt-4o-mini-2024-07-18')):
            raise ValueError('Unapproved generative model identity')
        choice = result['choices'][0]
        if choice.get('finish_reason') != 'stop':
            raise ValueError('Incomplete experience extraction')
        text = choice.get('message',{}).get('content')
        def reject_constant(value): raise ValueError('Non-finite generated JSON')
        payload = json.loads(text, parse_constant=reject_constant)
        return payload, {'logical_model':(self.model_plan or {}).get('logical_model','gpt-4o-mini-closeai'),'request_model':self.http.model,
                         'response_model':result['model'],'usage':result.get('usage',{}),
                         'prompt_sha256':digest(INSTRUCTION), 'recipe_fingerprint':self.fingerprint}

    def prepare(self, scope, session, request_id, payload_digest, canonical_sources):
        """Prepare additions for the SAME raw Add T1, outside a SQL transaction.

        The store owns its scope lease. Separate optional derived publication
        must never leave a pending frontier that blocks already acknowledged raw
        evidence. Raw, derived output and its job therefore publish atomically.
        """
        raw = sorted((m for m in canonical_sources if m.memory_type=='engineering_raw' and m.session_id==session),
                     key=lambda m:m.metadata['sequence'])
        units = [MemoryUnit(uid=m.logical_uid,raw_data={'text_content':m.text_content},
            metadata={'user_id':scope.user_id,'session_id':session,'request_id':m.request_id,
                      'role':m.role,'ts_ms':m.business_timestamp_ms,'sequence':m.metadata['sequence']}) for m in raw]
        old = [m.logical_uid for batch in self.store.state.iter_memories(scope) for m in batch
               if m.memory_type=='engineering_experience' and m.session_id==session
               and m.metadata.get('recipe_fingerprint')==self.fingerprint]
        empty = Mutation(delete_uids=tuple(old))
        if len(units)<2 or sum(len(u.raw_data['text_content']) for u in units)<256:
            return empty
        sources={u.uid:u for u in units}
        versions={m.logical_uid:m.version for m in raw}
        job=BuildJob.planned(scope,versions,self.fingerprint)
        checkpoint=self.ledger.reserve(scope,request_id,payload_digest)
        prepared=checkpoint.get('code_experience',{})
        if prepared.get('state') in ('inflight','failed'):
            self.store.state.record_event(scope,'derive','experience','UnknownOrFailedOutcome')
            return empty
        if prepared.get('state')!='prepared':
            base=EngineeringEpisodeBuilder().build(units,user_id=scope.user_id,session_id=session)
            checkpoint['code_experience']={'state':'inflight','input_versions':versions,'recipe':self.fingerprint}
            self.ledger.checkpoint(scope,request_id,payload_digest,checkpoint)
            try:
                payload,receipt=self.generate(self.distiller.prepare(base,sources),job.job_id)
                if self.model_plan and receipt.get('response_model') not in self.model_plan.get('accepted_response_models',[]):
                    raise ValueError('Unverified configured generative model identity')
                result=self.distiller.apply(base,sources,payload)
                # Persistent experience contains facets plus source references,
                # not a second copy of the full originating issue/request.
                result.episode.problem = result.episode.request = ''
                if result.episode.task_summary_claim is None:
                    result.episode.task_summary = result.episode.task_summary[:400]
                prepared={'state':'prepared','input_versions':versions,'recipe':self.fingerprint,
                          'episode':result.episode.to_memory_unit().raw_data,'audit':result.audit,'provider':receipt}
            except Exception as exc:
                checkpoint['code_experience']={'state':'failed','error_class':type(exc).__name__,
                                               'input_versions':versions,'recipe':self.fingerprint}
                self.ledger.checkpoint(scope,request_id,payload_digest,checkpoint)
                self.store.state.record_event(scope,'derive','experience',type(exc).__name__)
                return empty
            checkpoint['code_experience']=prepared
            self.ledger.checkpoint(scope,request_id,payload_digest,checkpoint)
        if prepared['input_versions']!=versions or prepared['recipe']!=self.fingerprint:
            self.store.state.record_event(scope,'derive','experience','SourceFrontierChanged')
            return empty  # Never bind completed extraction to different evidence.
        episode=self.decode(prepared['episode']);episode.validate_provenance(sources)
        if not prepared['audit'].get('accepted'):
            return Mutation(delete_uids=tuple(old),jobs=(BuildJob(job.job_id,versions,self.fingerprint,'completed',
                {'provider':prepared['provider'],'audit':prepared['audit']},()),))
        uid='engineering_experience:'+digest([scope.key,versions,self.fingerprint])
        memory=CanonicalMemory(uid,episode.retrieval_text(),raw_data=prepared,
            metadata={'recipe_fingerprint':self.fingerprint,'source_versions':versions,
                      'source_hashes':episode.source_hashes,'source_kind':'derived'},
            session_id=session,request_id=request_id,memory_tower='engineering',memory_level='experience',
            # The legacy named space is a frozen storage contract. Semantic
            # type/level and provenance distinguish experience from raw; adding
            # a new physical space must not change an existing DB manifest.
            memory_type='engineering_experience',spaces=('code_raw',))
        edges=tuple(Edge('code-support:'+digest([uid,source]),uid,source,'SUPPORTED_BY',
                        provenance={'source_version':version,'source_hash':episode.source_hashes[source]})
                    for source,version in versions.items())
        completed=BuildJob(job.job_id,versions,self.fingerprint,'completed',
                          {'provider':prepared['provider'],'audit':prepared['audit']},(uid,))
        return Mutation(memories=(memory,),delete_uids=tuple(old),edges=edges,jobs=(completed,))

    @staticmethod
    def decode(payload):
        return EngineeringEpisodeMemory.from_memory_unit(MemoryUnit(uid=payload['episode_id'],raw_data=payload))

    def overlay(self, user, episodes, sources):
        scope = self.store.scope(user)
        canonical = self.store.state.get_memories(scope,list(sources))
        derived = [m for batch in self.store.state.iter_memories(scope) for m in batch
                   if m.memory_type=='engineering_experience' and m.metadata.get('recipe_fingerprint')==self.fingerprint]
        by_session = {}
        for m in derived:
            try:
                versions = m.metadata['source_versions']
                if any(uid not in canonical or canonical[uid].version!=v for uid,v in versions.items()):
                    raise ValueError('Stale source revision')
                episode = self.decode(m.raw_data['episode'])
                episode.validate_provenance(sources)
                expected = next(e for e in episodes if e.session_id==episode.session_id)
                if set(expected.source_uids)!=set(versions): raise ValueError('Incomplete source frontier')
                by_session[episode.session_id]=episode
            except (KeyError,ValueError,StopIteration):
                self.store.state.record_event(scope,'hydrate','experience','InvalidProvenance')
        return [by_session.get(e.session_id,e) for e in episodes]

    def close(self):
        if self.http: self.http.close()
