"""HTTP-only native vLLM clients. No torch/transformers/model imports.

Wire translations reuse services/inference/.../vllm_bridge.py conventions.
The three numerical recipes are pinned separately from endpoint placement.
"""
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar,copy_context
from pathlib import PurePosixPath
import httpx
from .models import ModelError, RetrievalModels
from .retrieval_contract import MODELS, QUERY_PROMPT, recipe_hash


class NativeRetrievalModels(RetrievalModels):
    def __init__(self, settings, processor, transport=None):
        self.settings, self.processor = settings, processor
        self.urls = {'dense': settings.embedding_url.removesuffix('/').removesuffix('/v1'),
                     'sparse': settings.sparse_url.rstrip('/').removesuffix('/v1'),
                     'reranker': settings.reranker_url.rstrip('/').removesuffix('/v1')}
        self.keys = {'dense': settings.embedding_api_key_env, 'sparse': settings.sparse_api_key_env,
                     'reranker': settings.reranker_api_key_env}
        self.timeouts = {c: getattr(settings, c+'_timeout') for c in MODELS}
        self.semaphores = {c: threading.BoundedSemaphore(getattr(settings, c+'_concurrency')) for c in MODELS}
        self.http = httpx.Client(trust_env=False, timeout=30, transport=transport,
                                 limits=httpx.Limits(max_connections=12, max_keepalive_connections=6))
        self.parallel = ThreadPoolExecutor(max_workers=settings.dense_concurrency, thread_name_prefix='vllm-http')
        self.local = threading.local()
        self.health_lock = threading.RLock()
        self.health_state = {c: {'ready': False, 'reason': 'not_checked', 'checked_at': 0} for c in MODELS}
        self.verified = {c:False for c in MODELS}
        self.request_trace=ContextVar('chinasoft_retrieval_http_trace',default=None)

    def request(self, channel, method, path, payload=None, timeout=None):
        observation=self.request_trace.get();started=time.monotonic();status=None
        key = os.getenv(self.keys[channel]) if self.keys[channel] else None
        if self.keys[channel] and not key:
            raise ModelError(channel+'_credential', '检索服务凭据未配置')
        if not self.semaphores[channel].acquire(timeout=self.timeouts[channel]):
            raise ModelError(channel+'_busy', '检索模型并发已达上限，请稍后重试', retryable=True)
        try:
            r = self.http.request(method, self.urls[channel]+path, json=payload,
                                  headers={'Authorization': 'Bearer '+key} if key else {},
                                  timeout=timeout or self.timeouts[channel])
            status=r.status_code
            if r.status_code >= 400:
                raise ModelError(channel+'_http_'+str(r.status_code), 'vLLM 检索服务请求失败', retryable=r.status_code==429 or r.status_code>=500)
            return r.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ModelError(channel+'_connection', 'vLLM 检索服务连接、超时或 JSON 响应异常', retryable=True) from exc
        finally:
            if observation is not None and method=='POST':
                values=(payload or {}).get('text_2',(payload or {}).get('input',[]))
                observation.append({'channel':channel,'path':path,'items':len(values) if isinstance(values,list) else 1,
                                    'http_status':status,'elapsed_ms':round((time.monotonic()-started)*1000,3)})
            self.semaphores[channel].release()

    def probe(self, channel, force=False):
        with self.health_lock:
            old = self.health_state[channel]
            if not force and time.time()-old['checked_at'] < 15:
                return old
            m = MODELS[channel]
            try:
                v = self.request(channel, 'GET', '/version', timeout=10)
                models = self.request(channel, 'GET', '/v1/models', timeout=10).get('data', [])
                rows = [r for r in models if r.get('id') == m['alias']]
                if v.get('version') != m['version'] or len(rows) != 1 or PurePosixPath(rows[0].get('root', '')).name != m['revision']:
                    raise ValueError('native version/alias/pinned snapshot mismatch')
                metadata = self.request(channel, 'GET', '/mandol/metadata', timeout=10)
                cfg = metadata['vllm_config']
                if metadata.get('recipe_hash') != recipe_hash(channel) or metadata.get('artifact',{}).get('revision') != m['revision']:
                    raise ValueError('native artifact recipe attestation mismatch')
                if metadata['artifact'].get('weights_sha256') != m['weights_sha256']:
                    raise ValueError('native weight content differs from pinned recipe')
                mc = cfg['model_config']; hf = mc['hf_config']; pc = mc['pooler_config']
                dc = cfg['device_config']
                if (mc['dtype'] != 'bfloat16' or mc.get('quantization') is not None or mc['max_model_len'] != m['max_length']
                    or mc.get('runner_type', mc.get('runner')) != 'pooling' or dc.get('device_type') != 'cuda'):
                    raise ValueError('native GPU/precision/context/runner mismatch')
                if any(pc.get(k) != val for k, val in m['pooler'].items()):
                    raise ValueError('native pooler config mismatch')
                if channel == 'sparse' and (hf.get('architectures') != ['BertSpladeSparseEmbeddingModel']
                        or hf.get('cls_token_id', 'missing') is not None or hf.get('sep_token_id', 'missing') is not None
                        or hf.get('vocab_size') != 30522):
                    raise ValueError('SPLADE all-token vocabulary/pooling contract mismatch')
                if channel=='sparse' and (metadata['artifact'].get('tokenizer.json') != self.SPLADE_TOKENIZER or metadata['artifact'].get('config.json') != self.SPLADE_CONFIG):
                    raise ValueError('SPLADE tokenizer/config checksum mismatch')
                if channel=='reranker' and (pc.get('seq_pooling_type')!='CLS' or hf.get('architectures')!=['XLMRobertaForSequenceClassification']):
                    raise ValueError('BGE CLS classification contract mismatch')
                old = {'ready': self.verified[channel], 'identity_ready': True, 'reason': None if self.verified[channel] else 'inference_not_verified', 'checked_at': time.time(), 'model': m['model'],
                       'revision': m['revision'], 'vllm_version': m['version'], 'dtype': 'bfloat16', 'device': 'cuda',
                       'recipe_hash': recipe_hash(channel), 'attestation': 'native-server-config-and-snapshot-identity',
                       'pooling': {'dense': 'last-token-l2-normalized', 'sparse': self.SPLADE_POOLING,
                                   'reranker': 'cls-cross-encoder-sigmoid'}[channel]}
            except Exception as exc:
                self.verified[channel] = False
                old = {'ready': False, 'reason': getattr(exc, 'code', None) or 'native_contract_mismatch',
                       'checked_at': time.time(), 'model': m['model']}
            self.health_state[channel] = old
            return dict(old)

    def probe_sparse(self, force=False):
        return self.probe('sparse', force)

    def probe_all(self, force=False):
        result={}
        for channel in MODELS:
            health=self.probe(channel,force)
            if health.get('identity_ready') and not health['ready']:
                try:
                    if channel=='dense':self.dense(['ChinaSoft readiness'],'query')
                    elif channel=='sparse':self.sparse_batch(['ChinaSoft readiness'],'query')
                    else:self.rerank('camera',['A camera is in the cabinet.'])
                except ModelError:pass
                health=dict(self.health_state[channel])
            result[channel]=health
        return result

    def require(self, channel):
        if not self.probe(channel).get('identity_ready'):
            raise ModelError(channel+'_unavailable', 'vLLM '+channel+' 服务不可用或数值配方不符', retryable=True)

    def failed(self, channel):
        self.verified[channel]=False
        self.health_state[channel] = {'ready': False, 'reason': 'inference_failed', 'checked_at': 0}

    def succeeded(self,channel):
        self.verified[channel]=True
        self.health_state[channel].update(ready=True,reason=None)

    def ordered(self, channel, data, count, field):
        if data.get('model') != MODELS[channel]['alias']:
            raise ValueError('wrong served model')
        rows = data.get('data')
        if not isinstance(rows, list) or len(rows) != count:
            raise ValueError('partial inference response')
        if any(type(row.get('index')) is not int for row in rows):raise ValueError('invalid index')
        rows = sorted(rows, key=lambda r: r['index'])
        if [r['index'] for r in rows] != list(range(count)):raise ValueError('invalid indices')
        return [r[field] for r in rows]

    def dense(self, texts, mode='document'):
        self.require('dense'); start = time.monotonic(); result = []
        try:
            for text in texts:
                text = (QUERY_PROMPT if mode == 'query' else '')+text
                data = self.request('dense', 'POST', '/v1/embeddings', {
                    'model': MODELS['dense']['alias'], 'input': [text], 'encoding_format': 'float',
                    'truncate_prompt_tokens': 4096, 'truncation_side': 'right'})
                vector = self.ordered('dense', data, 1, 'embedding')[0]
                if (not isinstance(vector, list) or len(vector) != 1024 or
                    any(type(v) not in (int, float) or not math.isfinite(v) for v in vector) or abs(math.hypot(*vector)-1) > .02):
                    raise ValueError('Qwen dimension/normalization mismatch')
                result.append(vector)
            self.succeeded('dense')
            return result, (time.monotonic()-start)*1000
        except Exception as exc:
            self.failed('dense')
            if isinstance(exc, ModelError):raise
            raise ModelError('dense_schema', 'Qwen vLLM 输出不符合配方', retryable=True) from exc

    def sparse_batch(self, texts, mode='document'):
        self.require('sparse'); start = time.monotonic(); result=[]
        try:
            # Microbatch 1 retains the qualified all-token pooling resource bound.
            for text in texts:
                data=self.request('sparse', 'POST', '/pooling', {'model': MODELS['sparse']['alias'],
                    'input': [text], 'task': 'embed', 'encoding_format': 'float', 'add_special_tokens': True,
                    'truncate_prompt_tokens': 512, 'truncation_side': 'right', 'use_activation': False})
                vector=self.ordered('sparse', data, 1, 'data')[0]
                if not isinstance(vector, list) or len(vector)!=30522 or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in vector):
                    raise ValueError('invalid SPLADE vocabulary activation')
                sparse={i:float(v) for i,v in enumerate(vector) if v>0}
                if not sparse:raise ValueError('empty SPLADE activation')
                result.append(sparse)
            self.local.sparse_ms=(time.monotonic()-start)*1000
            self.local.sparse_server_ms=0  # Native HTTP does not report model-only latency.
            self.succeeded('sparse')
            return result
        except Exception as exc:
            self.failed('sparse')
            if isinstance(exc, ModelError):raise
            raise ModelError('sparse_schema', 'SPLADE vLLM 输出不符合词表/池化协议', retryable=True) from exc

    def encode(self, texts, mode):
        if mode not in ('query', 'document') or (mode=='query' and len(texts)!=1):raise ValueError('explicit encode mode required')
        texts=[t[:4000] or '[空文本]' for t in texts]
        future=self.parallel.submit(copy_context().run,self.dense,texts,mode)
        try:sparse=self.sparse_batch(texts,mode)
        except Exception:
            try:future.result()
            except Exception:pass
            raise
        dense,dense_ms=future.result()
        self.local.timings={'dense_encode_ms':round(dense_ms,2), 'sparse_encode_ms':round(self.local.sparse_ms,2),
                            'sparse_mode':mode, 'serving':'native-vllm-gpu-http'}
        return list(zip(dense,sparse,strict=True))

    def rerank(self, query, documents):
        if not documents:return []
        self.require('reranker')
        try:
            data=self.request('reranker', 'POST', '/score', {'model':MODELS['reranker']['alias'],
                 'text_1':query, 'text_2':documents, 'truncate_prompt_tokens':4096, 'truncation_side':'right', 'use_activation':True})
            scores=self.ordered('reranker',data,len(documents),'score')
            if any(type(v) not in (float,int) or not math.isfinite(v) or not 0<=v<=1 for v in scores):raise ValueError('invalid sigmoid scores')
            self.succeeded('reranker')
            return sorted(enumerate(scores),key=lambda r:(-r[1],r[0]))
        except Exception as exc:
            self.failed('reranker')
            if isinstance(exc,ModelError):raise
            raise ModelError('reranker_schema','BGE vLLM 重排输出不合法',retryable=True) from exc
