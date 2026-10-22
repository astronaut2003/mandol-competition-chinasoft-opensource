"""Bounded real OpenAI-compatible generation, embedding and BAAI reranking."""
import asyncio, base64, hashlib, json, math, os, re, threading, time
import httpx
from jsonschema import Draft202012Validator
from mandol_substrate.state.models import digest, canonical_json
from mandol_competition.textual.core.user_profile import strict_json
from mandol_competition.textual.core.hlm_production import ProductionHLMProvider
from .store import now
from .profile_policy import C_GUARD,C_RELATIONS,A_GUARD


class ModelError(Exception):
    def __init__(self, code, message, *, retryable=False, unknown=False):
        self.code, self.message, self.retryable, self.unknown = code, message, retryable, unknown
        super().__init__(message)


class GenerationClient:
    def __init__(self, settings, store, transport=None, retry_sleep=time.sleep):
        self.settings, self.store = settings, store
        self.transport=transport
        self.slots = threading.BoundedSemaphore(settings.llm_max_concurrency)
        self.background_slots = threading.BoundedSemaphore(max(1,settings.llm_max_concurrency-1))
        self.client = httpx.Client(timeout=settings.llm_timeout, trust_env=False, transport=transport,
                                  limits=httpx.Limits(max_connections=4, max_keepalive_connections=4))
        self.retry_sleep = retry_sleep
        self.local = threading.local()
        from contextvars import ContextVar
        self.stream_model = ContextVar('chinasoft_response_model', default=None)
        self.stream_metadata = ContextVar('chinasoft_generation_metadata',default=None)
        self.account_context = ContextVar('chinasoft_generation_account', default=None)

    def charge(self,job_id=None):
        account=self.account_context.get()
        from .store import AppError
        try:
            with self.store.state.transaction() as db:
                if not account and job_id:
                    row=db.execute('SELECT owner FROM cs_jobs WHERE id=?',(job_id,)).fetchone()
                    found=self.store.access.account_for(db,row['owner']) if row else None
                    account=found['id'] if found else None
                if account:self.store.access.consume(db,account,'model_per_day',86400000)
        except AppError as exc:raise ModelError(exc.code,exc.message) from exc

    def headers(self):
        key = os.getenv(self.settings.llm_api_key_env)
        if not key: raise ModelError('model_not_configured', 'DeepSeek API 密钥未配置')
        return {'Authorization': 'Bearer ' + key}

    def payload(self, messages, vision=False, schema=None, stream=False, reasoning_mode='fast'):
        msgs = [dict(m) for m in messages]
        body = {'model': self.settings.vlm_request_model if vision else self.settings.llm_request_model,
                'messages': msgs, 'max_tokens': self.settings.llm_max_tokens, 'temperature': 0, 'stream': stream}
        if 'deepseek' in self.settings.llm_base_url:
            body['thinking'] = {'type':'disabled'}
        capabilities=self.settings.reasoning_capabilities(vision)
        if reasoning_mode not in capabilities['modes']:raise ModelError('reasoning_unsupported','当前生成端点尚未通过此推理模式的协议配置，请选择快速模式')
        body.update(capabilities['mapping'][reasoning_mode])
        if reasoning_mode!='fast':body.pop('temperature',None)
        if stream and capabilities['protocol']=='deepseek':body['stream_options']={'include_usage':True}
        if schema:
            msgs.insert(0, {'role':'system','content':'Return ONLY JSON matching this JSON schema. All nullable fields must be present. '+canonical_json(schema)})
            mode = self.settings.json_output_mode
            if mode == 'json_object': body['response_format'] = {'type':'json_object'}
            elif mode == 'json_schema': body['response_format'] = {'type':'json_schema','json_schema':{'name':'memory_extraction','strict':True,'schema':schema}}
        return body

    def validate(self, data, vision=False, schema=None):
        contract = self.settings.contract(vision)
        if data.get('model') not in contract['accepted_response_models']:
            raise ModelError('model_identity', '返回模型 ID 不符合已配置的模型身份')
        try:
            choice = data['choices'][0]; text = choice['message']['content']
        except (KeyError, IndexError, TypeError): raise ModelError('invalid_response', '模型响应结构不完整')
        if choice.get('finish_reason') != 'stop':
            raise ModelError('incomplete_response', '生成结果被截断或未正常完成')
        if not isinstance(text, str) or not text.strip(): raise ModelError('empty_response', '模型返回空内容')
        if schema:
            try:
                parsed = strict_json(text); Draft202012Validator(schema).validate(parsed)
            except Exception as exc:
                raise ModelError('invalid_json', '模型结构化输出不符合 Schema，已拒绝写入高阶记忆') from exc
        return {'status':'PASS','response_model':data['model'],'finish_reason':'stop','content':text,
                'usage':data.get('usage',{}),'model_contract':contract}

    def complete(self, messages, *, schema=None, vision=False, cache=True, max_tokens=None, timeout=None, attempts=None, foreground=False,reasoning_mode='fast'):
        payload = self.payload(messages, vision, schema,reasoning_mode=reasoning_mode)
        if max_tokens is not None: payload['max_tokens'] = min(self.settings.llm_max_tokens, max_tokens)
        job_id = None if foreground else getattr(self.local, 'job_id', None) or getattr(self, 'active_build_job', None)
        key = 'call_' + digest([job_id, self.settings.generation_fingerprint, payload])
        if cache:
            with self.store.state.transaction() as db:
                existing = db.execute('SELECT * FROM cs_calls WHERE id=?', (key,)).fetchone()
                if existing and existing['state'] == 'completed': return json.loads(existing['response'])
                if existing and existing['state'] in ('running','unknown'):
                    raise ModelError('unknown_remote', '此前模型调用结果未知，需在任务页明确确认后重试', unknown=True)
                db.execute("INSERT INTO cs_calls(id,job_id,state,created_at,updated_at) VALUES (?,?,'reserved',?,?) ON CONFLICT(id) DO UPDATE SET state='reserved',updated_at=excluded.updated_at", (key, job_id, now(), now()))
        started = time.monotonic()
        background_acquired = False
        try:
            if job_id:
                background_acquired = self.background_slots.acquire(timeout=self.settings.llm_timeout)
                if not background_acquired: raise ModelError('model_busy','后台模型并发已满，请稍后重试',retryable=True)
            if not self.slots.acquire(timeout=timeout or self.settings.llm_timeout):
                raise ModelError('model_busy', '模型并发已满，请稍后重试', retryable=True)
            try:
                if cache:
                    with self.store.state.transaction() as db:
                        db.execute("UPDATE cs_calls SET state='running',updated_at=? WHERE id=?",(now(),key))
                limit = attempts or self.settings.generation_retry_policy
                for attempt in range(limit):
                    self.charge(job_id)
                    try:
                        r = self.client.post(self.settings.llm_base_url.rstrip('/') + '/chat/completions',
                                             headers=self.headers(), json=payload, timeout=timeout or self.settings.llm_timeout)
                    except httpx.TimeoutException as exc:
                        raise ModelError('model_timeout', '模型请求超时，远程结果未知', unknown=True) from exc
                    except httpx.ConnectError as exc:
                        raise ModelError('model_connection', '模型连接失败', retryable=True) from exc
                    except httpx.TransportError as exc:
                        raise ModelError('model_network', '模型连接在响应期间中断，远程结果未知', unknown=True) from exc
                    if r.status_code == 429 or r.status_code >= 500:
                        if attempt + 1 < limit:
                            try: delay = min(15, max(0.2, float(r.headers.get('retry-after', 0))))
                            except ValueError: delay = 0
                            self.retry_sleep(max(delay, 0.5 * 2 ** attempt)); continue
                        raise ModelError('rate_limited' if r.status_code == 429 else 'model_server', '模型服务限流或暂时不可用', retryable=True)
                    if r.status_code >= 400: raise ModelError('model_http_' + str(r.status_code), '模型请求被端点拒绝，请检查配置和权限')
                    if len(r.content) > 1024 * 1024: raise ModelError('oversized_response', '模型响应超过安全预算')
                    try: data = r.json()
                    except ValueError as exc: raise ModelError('invalid_response', '模型未返回 JSON 响应') from exc
                    result = self.validate(data, vision, schema)
                    result['reasoning']={'mode':reasoning_mode,'parameters':self.settings.reasoning_capabilities(vision)['mapping'][reasoning_mode],
                        'observed_chars':len(data['choices'][0]['message'].get('reasoning_content') or '')}
                    details=(result.get('usage') or {}).get('completion_tokens_details') or {}
                    observed=bool(result['reasoning']['observed_chars'] or details.get('reasoning_tokens'))
                    result['reasoning']['thinking_observed']=observed
                    if reasoning_mode!='fast' and not observed:raise ModelError('reasoning_not_observed','端点没有返回可验证的思考模式使用量，本轮未标记为该模式成功')
                    result['latency_ms'] = round((time.monotonic() - started) * 1000, 2)
                    if cache:
                        with self.store.state.transaction() as db:
                            db.execute("UPDATE cs_calls SET state='completed',response=?,updated_at=? WHERE id=?", (canonical_json(result), now(), key))
                    self.store.metric('generation', result['latency_ms'])
                    return result
            finally: self.slots.release()
        except ModelError as exc:
            if cache:
                with self.store.state.transaction() as db:
                    db.execute('UPDATE cs_calls SET state=?,error_code=?,updated_at=? WHERE id=?', ('unknown' if exc.unknown else 'failed', exc.code, now(), key))
            raise
        finally:
            if background_acquired: self.background_slots.release()

    async def stream(self, messages, vision=False,reasoning_mode='fast'):
        self.stream_model.set(None)
        body = self.payload(messages, vision, stream=True,reasoning_mode=reasoning_mode); headers = self.headers()
        metadata={'mode':reasoning_mode,'parameters':self.settings.reasoning_capabilities(vision)['mapping'][reasoning_mode],
                  'usage':{},'reasoning_chars':0,'thinking_observed':False}
        self.stream_metadata.set(metadata)
        # One shared semaphore across async Chat and sync high-order workers.
        deadline=time.monotonic()+self.settings.llm_timeout
        while not self.slots.acquire(blocking=False):
            if time.monotonic()>=deadline:raise ModelError('model_busy','模型并发已满，请稍后重试',retryable=True)
            await asyncio.sleep(0.05)
        done, finished, actual = False, False, None
        try:
            self.charge()
            async with asyncio.timeout(self.settings.llm_timeout):
                async with httpx.AsyncClient(timeout=self.settings.llm_timeout, trust_env=False,transport=self.transport) as client:
                    async with client.stream('POST', self.settings.llm_base_url.rstrip('/') + '/chat/completions', headers=headers, json=body) as r:
                        if r.status_code >= 400: raise ModelError('model_http_' + str(r.status_code), '模型流式接口暂时不可用')
                        async for line in r.aiter_lines():
                            if not line.startswith('data:'): continue
                            value = line[5:].strip()
                            if value == '[DONE]': done = True; break
                            try: chunk = json.loads(value)
                            except ValueError as exc: raise ModelError('invalid_stream', '流式响应格式不合法') from exc
                            actual = chunk.get('model', actual)
                            if actual not in self.settings.contract(vision)['accepted_response_models']:
                                raise ModelError('model_identity', '流式返回模型身份不符')
                            self.stream_model.set(actual)
                            if isinstance(chunk.get('usage'),dict):metadata['usage']=chunk['usage']
                            for choice in chunk.get('choices', []):
                                reasoning=choice.get('delta',{}).get('reasoning_content')
                                if isinstance(reasoning,str):
                                    metadata['reasoning_chars']+=len(reasoning)
                                    metadata['thinking_observed']=bool(metadata['reasoning_chars'])
                                if metadata['reasoning_chars']>240000:raise ModelError('reasoning_budget','模型思考输出超过安全预算')
                                text = choice.get('delta', {}).get('content')
                                if isinstance(text, str) and text: yield text
                                if choice.get('finish_reason'):
                                    if choice['finish_reason'] != 'stop': raise ModelError('incomplete_response', '流式回答被截断')
                                    finished = True
            if not done or not finished: raise ModelError('incomplete_response', '流式连接未正常结束')
            metadata['thinking_observed']=bool(metadata['reasoning_chars'] or (metadata['usage'].get('completion_tokens_details') or {}).get('reasoning_tokens'))
            if reasoning_mode!='fast' and not metadata['thinking_observed']:raise ModelError('reasoning_not_observed','端点没有返回可验证的思考模式使用量，本轮未标记为该模式成功')
            if reasoning_mode=='fast' and metadata['thinking_observed']:raise ModelError('reasoning_mismatch','端点未遵循关闭思考的请求，请检查模型兼容性')
        except (httpx.TimeoutException,TimeoutError) as exc: raise ModelError('model_timeout', '流式生成超时，已保存部分回答', unknown=True) from exc
        except httpx.TransportError as exc: raise ModelError('model_network', '生成连接中断，已保存部分回答', unknown=True) from exc
        finally: self.slots.release()

    def image_block(self, media):
        data = (self.settings.data_dir / media['path']).read_bytes()
        return {'type':'image_url','image_url':{'url':'data:' + media['mime'] + ';base64,' + base64.b64encode(data).decode(), 'detail':'high'}}

    def close(self): self.client.close()


class HLMProvider(ProductionHLMProvider):
    def __init__(self, client, version3=False): self.client,self.version3 = client,version3

    def __call__(self, messages):
        fmt = self.response_format(messages, 'required')
        if json.loads(messages[-1]['content']).get('stage')=='A':
            messages=[{'role':'system','content':A_GUARD},*messages]
            if self.version3:
                from .highlevel import PROFILE_QUALIFIER_GUARD
                messages.insert(0,{'role':'system','content':PROFILE_QUALIFIER_GUARD})
                item=fmt['json_schema']['schema']['properties']['candidates']['items']['properties']['item']
                item['allOf']=[{'if':{'properties':{'lifetime':{'const':'conditional'}}},'then':{'anyOf':[{'properties':{'condition':{'type':'string','minLength':1}}},{'properties':{'scope':{'type':'string','minLength':1}}}]}},{'if':{'properties':{'lifetime':{'const':'temporary'}}},'then':{'properties':{'time_qualifier':{'type':'string','minLength':1}}}}]
            properties=fmt['json_schema']['schema']['properties']['candidates']['items']['properties']['item']['properties']
            for field in ('valid_from','valid_to'):
                properties[field]['pattern']=r'^\d{4}-\d{2}-\d{2}$'
        if json.loads(messages[-1]['content']).get('stage')=='C':
            messages=[{'role':'system','content':C_GUARD},*messages]
            fmt['json_schema']['schema']['allOf']=[{'if':{'properties':{'op':{'const':op}}},'then':{'properties':{'value_relation':{'const':relation}}}} for op,relation in C_RELATIONS.items()]
        return self.client.complete(messages, schema=fmt['json_schema']['schema'])


class RetrievalModels:
    SPLADE_MODEL='naver/splade-v3'
    SPLADE_REVISION='fdfeceb91d7b9de7985b38addd3ba9f53a59a355'
    SPLADE_POOLING='max-log1p-relu-masked-full-token-v1'
    SPLADE_WEIGHTS='e59283d1f2db78da12abc9dc9ffd499f4a58c66652ad746e4d9bd01adc383077'
    SPLADE_TOKENIZER='2fc687b11de0bc1b3d8348f92e3b49ef1089a621506c7661fbf3248fcd54947e'
    SPLADE_CONFIG='460759cb5e43a454a755d25d293d0ca0d6ad67188cf74eb0525ce13ed7c7b433'

    def __init__(self, settings, processor, transport=None):
        from concurrent.futures import ThreadPoolExecutor
        self.settings, self.processor = settings, processor
        self.http = httpx.Client(timeout=20, trust_env=False, transport=transport,
                                 limits=httpx.Limits(max_connections=4, max_keepalive_connections=4))
        self.slots = threading.BoundedSemaphore(1)
        self.sparse_slots = threading.BoundedSemaphore(1)
        self.parallel = ThreadPoolExecutor(max_workers=2,thread_name_prefix='chinasoft-encode')
        self.sparse_health={'ready':False,'reason':'not_checked','checked_at':0}
        self.local=threading.local()

    def post(self, base, path, key_env, payload):
        key = os.getenv(key_env)
        if not key: raise ModelError('retrieval_not_configured', '检索模型 API 密钥未配置')
        with self.slots:
            try: r = self.http.post(base.rstrip('/') + path, headers={'Authorization':'Bearer ' + key}, json=payload)
            except httpx.TransportError as exc: raise ModelError('retrieval_connection', '检索模型不可用', retryable=True) from exc
        if r.status_code >= 400: raise ModelError('retrieval_http_' + str(r.status_code), '检索模型暂时不可用', retryable=r.status_code == 429 or r.status_code >= 500)
        try:return r.json()
        except ValueError as exc:raise ModelError('retrieval_schema','检索服务未返回有效 JSON') from exc

    def probe_sparse(self, force=False):
        if not self.settings.sparse_url:
            self.sparse_health={'ready':False,'reason':'explicit_lexical_degradation','checked_at':time.time()}
            return self.sparse_health
        if not force and time.time()-self.sparse_health['checked_at']<15:return self.sparse_health
        try:
            response=self.http.get(self.settings.sparse_url.rstrip('/')+'/metadata',timeout=2)
            response.raise_for_status();meta=response.json()
            expected={'ready':True,'model':self.SPLADE_MODEL,'revision':self.SPLADE_REVISION,
                      'pooling':self.SPLADE_POOLING,'vocabulary_size':30522,'max_length':512,
                      'query_mode':'full','document_mode':'full','weights_sha256':self.SPLADE_WEIGHTS,'tokenizer_sha256':self.SPLADE_TOKENIZER,'config_sha256':self.SPLADE_CONFIG}
            if any(meta.get(k)!=v for k,v in expected.items()):raise ValueError('identity')
            if not re.fullmatch(r'[a-f0-9]{64}',meta.get('weights_sha256','')):raise ValueError('checksum')
            self.sparse_health={**meta,'reason':None,'checked_at':time.time()}
        except Exception:
            self.sparse_health={'ready':False,'reason':'splade_unavailable_or_contract_mismatch','checked_at':time.time()}
        return self.sparse_health

    def sparse_batch(self, texts, mode='document'):
        if not self.settings.sparse_url:
            if not self.settings.allow_retrieval_degraded:
                raise ModelError('splade_not_configured','SPLADE 未配置；请配置真实服务或显式允许降级',retryable=True)
            result=[]
            for text in texts:
                weights={}
                for term in self.processor.query(text).split():
                    index=int.from_bytes(hashlib.sha256(term.encode()).digest()[:4],'big')
                    weights[index]=weights.get(index,0)+1.0
                result.append({i:1+math.log(v) for i,v in weights.items()})
            return result
        if not self.probe_sparse()['ready']:raise ModelError('splade_unavailable','SPLADE 身份或服务不可用，三路检索未就绪',retryable=True)
        result=[];elapsed=0.0;server=0.0
        with self.sparse_slots:
            for offset in range(0,len(texts),4):
                batch=list(texts[offset:offset+4]);start=time.monotonic()
                try:
                    response=self.http.post(self.settings.sparse_url.rstrip('/')+'/v1/sparse',
                        json={'model':self.SPLADE_MODEL,'input':batch,'mode':mode})
                    response.raise_for_status();data=response.json()
                    if data.get('model')!=self.SPLADE_MODEL or data.get('revision')!=self.SPLADE_REVISION or data.get('pooling')!=self.SPLADE_POOLING or data.get('mode')!=mode:
                        raise ValueError('sparse identity')
                    rows=data['data']
                    if len(rows)!=len(batch) or [r['index'] for r in rows]!=list(range(len(batch))):raise ValueError('partial sparse')
                    for row in rows:
                        sparse={}
                        for key,value in row['sparse'].items():
                            index=int(key)
                            if str(index)!=str(key) or not 0<=index<30522 or isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
                                raise ValueError('sparse schema')
                            sparse[index]=float(value)
                        if not sparse:raise ValueError('empty sparse')
                        result.append(sparse)
                    server+=float(data.get('elapsed_ms',0))
                except Exception as exc:
                    self.sparse_health={'ready':False,'reason':'splade_inference_failed','checked_at':0}
                    raise ModelError('splade_inference','真实 SPLADE 编码失败，投影未标记就绪',retryable=True) from exc
                elapsed+=(time.monotonic()-start)*1000
        self.local.sparse_ms=elapsed;self.local.sparse_server_ms=server
        return result

    def sparse(self,text):return self.sparse_batch([text],mode='query')[0]

    def dense(self,texts):
        start=time.monotonic()
        data=self.post(self.settings.embedding_url,'/embeddings',self.settings.embedding_api_key_env,
                       {'model':self.settings.embedding_model,'input':list(texts)})
        if data.get('model',self.settings.embedding_model)!=self.settings.embedding_model:raise ModelError('embedding_identity','Embedding 模型身份不符')
        rows=sorted(data.get('data',[]),key=lambda x:x['index'])
        if len(rows)!=len(texts) or [r['index'] for r in rows]!=list(range(len(texts))):raise ModelError('embedding_partial','Embedding 返回不完整')
        vectors=[]
        for row in rows:
            vector=row['embedding']
            if len(vector)!=self.settings.embedding_dimension or any(not math.isfinite(float(v)) for v in vector) or not any(vector):raise ModelError('embedding_schema','Embedding 维度或数值不合法')
            vectors.append(vector)
        return vectors,(time.monotonic()-start)*1000

    def encode(self,texts,mode):
        texts=[text[:4000] or '[空文本]' for text in texts]
        future=self.parallel.submit(self.dense,texts)
        try:sparse=self.sparse_batch(texts,mode)
        except Exception:
            # Join the bounded in-flight request; never leak an orphan task.
            try:future.result()
            except Exception:pass
            raise
        dense,dense_ms=future.result()
        self.local.timings={'dense_encode_ms':round(dense_ms,2),'sparse_encode_ms':round(getattr(self.local,'sparse_ms',0),2),
                            'sparse_server_ms':round(getattr(self.local,'sparse_server_ms',0),2),'sparse_mode':mode}
        return list(zip(dense,sparse,strict=True))

    def encode_documents(self,texts):return self.encode(list(texts),'document')
    def encode_query(self,text):return self.encode([text],'query')[0]

    def rerank(self, query, documents):
        if not documents:return []
        data=self.post(self.settings.reranker_url,'/rerank',self.settings.reranker_api_key_env,
                       {'model':self.settings.reranker_model,'query':query,'documents':documents,'top_n':len(documents),'return_documents':False})
        rows=data.get('results',[]);seen=set()
        for row in rows:
            if type(row.get('index')) is not int or not 0<=row['index']<len(documents) or row['index'] in seen or not math.isfinite(float(row.get('relevance_score',float('nan')))):
                raise ModelError('reranker_schema','Reranker 返回非法排序')
            seen.add(row['index'])
        if len(seen)!=len(documents):raise ModelError('reranker_partial','Reranker 返回不完整')
        return [(row['index'],float(row['relevance_score'])) for row in rows]

    def close(self):
        self.parallel.shutdown(wait=True,cancel_futures=True);self.http.close()
