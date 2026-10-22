"""Outbox-owned scheduling of the existing four-tower producer.

Accepted producer receipts survive retries. A rejected preparation never becomes
a frozen successful root. Other sources can progress after a failed input; the
application job ledger, rather than a global pending-root scan, owns recovery.
"""
from mandol_competition.textual.core.hlm_production import CanonicalHighLevelMemory,ProductionHLMRecipe
from dataclasses import dataclass,replace,asdict
from mandol_substrate.state.models import digest
from .profile_policy import C_GUARD,A_GUARD
from mandol_competition.textual.core.profile_discourse import DiscourseRecipe
from .models import ModelError
from mandol_competition.textual.core.profile_discourse import empty_frame
import json, logging

@dataclass(frozen=True)
class ChinaSoftDiscourseRecipe(DiscourseRecipe):
    @property
    def skip_context_only(self):return True
    @property
    def session_scoped_frame(self):return True
    @property
    def fingerprint(self):return digest([super().fingerprint,C_GUARD,A_GUARD,'session-frame-and-context-skips-v12-semantics'])

@dataclass(frozen=True)
class ChinaSoftHLMRecipe(ProductionHLMRecipe):
    @property
    def profile_recipe(self):
        return ChinaSoftDiscourseRecipe(**dict(asdict(super().profile_recipe),version='chinasoft-profile-v2'))

    @property
    def prompt_map(self):
        prompts=dict(super().prompt_map)
        prompts['relation']+=' Only emit predicates compatible with supplied endpoint kinds; MEMBER_OF requires PERSON -> ORGANIZATION, never PROJECT. Omit unsupported relations; do not repair endpoint identity.'
        return prompts

    @property
    def fingerprint(self):return digest([super().fingerprint,C_GUARD,A_GUARD,'source-version-aware-discourse-frame-v2'])


PROFILE_QUALIFIER_GUARD = ('For lifetime conditional, condition or scope MUST be a nonempty source-grounded string. '
    'For lifetime temporary, time_qualifier MUST preserve the exact temporal phrase. '
    'Use stable only for an explicitly stable preference with no missing condition. '
    'Preserve corrections and negation. Never turn attachment, code, OCR or assistant content into personal attributes.')

@dataclass(frozen=True)
class ConversationRecipeV3(ChinaSoftHLMRecipe):
    @property
    def fingerprint(self):return digest([super().fingerprint,PROFILE_QUALIFIER_GUARD,'independent-audited-outputs-v3'])


class ChinaSoftHighLevelMemory(CanonicalHighLevelMemory):
    def _frame(self,scope,session):
        rows=self.state._rows("SELECT input_versions,progress FROM build_jobs WHERE scope=? AND recipe_fingerprint=? AND status='completed' AND json_extract(progress,'$.discourse_session_id') IS ? ORDER BY commit_seq DESC LIMIT 1",(scope.key,self.recipe.fingerprint,session))
        if not rows:return empty_frame()
        versions=json.loads(rows[0]['input_versions']);sources=self.state.get_memories(scope,list(versions))
        if any(u not in sources or not sources[u].active or sources[u].version!=v for u,v in versions.items()):return empty_frame()
        return json.loads(rows[0]['progress']).get('frame',empty_frame())

    def item_metadata(self, scope, kind, row, uid, root):
        return {**super().item_metadata(scope,kind,row,uid,root),
                'generation':self.provider.client.settings.contract(),
                'generation_fingerprint':self.provider.client.settings.generation_fingerprint}

    def build(self, scope, source_uids):
        with self.gates.lease(scope, write=True):
            return self._ordered_build(scope, source_uids)

    def prepare(self, scope, sources, worker, root):
        prepared = super().prepare(scope, sources, worker, root)
        if not prepared.get('errors'): return prepared
        logging.getLogger(__name__).warning('Audited HLM rejected stages: %s',prepared['errors'])
        job_id = getattr(self.provider.client, 'active_build_job', None)
        calls = self.state._rows('SELECT state,error_code FROM cs_calls WHERE job_id=?', (job_id,))
        unknown = any(c['state']=='unknown' for c in calls)
        transport_failure = any(c['state']=='failed' for c in calls)
        raise ModelError('derivation_rejected' if unknown or transport_failure else 'semantic_rejected',
                         '高阶输出未通过完整结构或来源审计，基础记忆完整保留',
                         unknown=unknown, retryable=any(c['error_code'] in ('rate_limited','model_server','model_connection') for c in calls))


class ConversationMemoryV3(ChinaSoftHighLevelMemory):
    def prepare(self,scope,sources,worker,root):
        # The existing producer validates every item and empties rejected towers.
        # Keep only independently accepted outputs; never restore a rejected row.
        prepared=CanonicalHighLevelMemory.prepare(self,scope,sources,worker,root)
        if prepared.get('errors'):
            logging.getLogger(__name__).warning('V3 independent output rejections: %s',prepared['errors'])
        return prepared
