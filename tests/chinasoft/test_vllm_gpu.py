"""Protocol/security fault injection, not a substitute for real GPU tests."""
from copy import deepcopy
from dataclasses import replace
import json,math,time
import httpx,pytest
from mandol_competition.chinasoft.config import Settings
from mandol_competition.chinasoft.models import ModelError,RetrievalModels
from mandol_competition.chinasoft.native_retrieval import NativeRetrievalModels
from mandol_competition.chinasoft.retrieval_contract import MODELS,QUERY_PROMPT,recipe_hash
from mandol_competition.chinasoft.service import Platform


def config(tmp_path,**kw):
    kw.setdefault('sparse_url','http://127.0.0.1:20636')
    return Settings(data_dir=tmp_path,retrieval_backend='vllm',embedding_model='Qwen/Qwen3-Embedding-0.6B',
                    embedding_api_key_env='',reranker_api_key_env='',**kw)


def metadata(channel):
    m=MODELS[channel];hf={'architectures':['XLMRobertaForSequenceClassification'] if channel=='reranker' else ['Qwen3ForCausalLM']}
    if channel=='sparse':hf={'architectures':['BertSpladeSparseEmbeddingModel'],'cls_token_id':None,'sep_token_id':None,'vocab_size':30522}
    return {'recipe_hash':recipe_hash(channel),'artifact':{'revision':m['revision'],'weights_sha256':m['weights_sha256'],
            'tokenizer.json':RetrievalModels.SPLADE_TOKENIZER,'config.json':RetrievalModels.SPLADE_CONFIG},
            'vllm_config':{'device_config':{'device_type':'cuda'},'model_config':{'runner_type':'pooling','dtype':'bfloat16',
                'quantization':None,'max_model_len':m['max_length'],'hf_config':hf,
                'pooler_config':{'seq_pooling_type':'CLS',**m['pooler']}}}}


def native_transport(channel,requests,mutate=None,response_mutate=None):
    def handle(request):
        body=json.loads(request.content) if request.content else None;requests.append((request.url.path,body))
        m=MODELS[channel]
        if request.url.path=='/version':data={'version':m['version']}
        elif request.url.path=='/v1/models':data={'data':[{'id':m['alias'],'root':'/cache/snapshots/'+m['revision']}]}
        elif request.url.path=='/mandol/metadata':
            data=metadata(channel)
            if mutate:mutate(data)
        elif request.url.path=='/pooling':data={'model':m['alias'],'data':[{'index':0,'data':[2.0]+[0.0]*30521}]}
        elif request.url.path=='/v1/embeddings':data={'model':m['alias'],'data':[{'index':0,'embedding':[1.0]+[0.0]*1023}]}
        elif request.url.path=='/score':data={'model':m['alias'],'data':[{'index':i,'score':.9-i*.2} for i in range(len(body['text_2']))]}
        else:raise AssertionError(request.url.path)
        if response_mutate and request.method=='POST':response_mutate(data)
        return httpx.Response(200,json=data)
    return httpx.MockTransport(handle)


def test_native_query_prompt_once_and_document_unprompted(tmp_path):
    calls=[];client=NativeRetrievalModels(config(tmp_path),None,native_transport('dense',calls))
    try:
        client.dense(['needle'],'query');client.dense(['needle'],'document')
        bodies=[b for p,b in calls if p=='/v1/embeddings']
        assert bodies[0]['input']==[QUERY_PROMPT+'needle'] and bodies[1]['input']==['needle']
        assert bodies[0]['truncate_prompt_tokens']==4096 and bodies[0]['truncation_side']=='right'
    finally:client.close()


def test_sparse_pooling_exact_wire_and_qdrant_vocabulary(tmp_path):
    calls=[];client=NativeRetrievalModels(config(tmp_path),None,native_transport('sparse',calls))
    try:
        assert client.sparse_batch(['document'])==[{0:2.0}] and client.sparse('query')=={0:2.0}
        body=[b for p,b in calls if p=='/pooling'][-1]
        assert body['task']=='embed' and body['add_special_tokens'] and body['use_activation'] is False
        assert body['truncate_prompt_tokens']==512 and body['input']==['query']
        assert not any(p=='/v1/sparse' for p,b in calls)
    finally:client.close()


@pytest.mark.parametrize('change',[
    lambda d:d['vllm_config']['device_config'].update(device_type='cpu'),
    lambda d:d['vllm_config']['model_config'].update(dtype='float32'),
    lambda d:d['vllm_config']['model_config']['hf_config'].update(cls_token_id=101),
    lambda d:d['vllm_config']['model_config']['hf_config'].update(sep_token_id=102),
    lambda d:d['vllm_config']['model_config']['hf_config'].update(vocab_size=999),
    lambda d:d['vllm_config']['model_config']['pooler_config'].update(use_activation=True),
    lambda d:d['artifact'].update(weights_sha256='0'*64),
    lambda d:d.update(recipe_hash='wrong')])
def test_refuse_incompatible_sparse_recipe(tmp_path,change):
    client=NativeRetrievalModels(config(tmp_path),None,native_transport('sparse',[],mutate=change))
    try:
        assert not client.probe_sparse(force=True)['ready']
        with pytest.raises(ModelError):client.sparse('query')
    finally:client.close()


@pytest.mark.parametrize('bad',[[],[0.0]*30522,[-1.0]+[0.0]*30521,[float('inf')]+[0.0]*30521])
def test_refuse_invalid_sparse_response_and_mark_unready(tmp_path,bad):
    def mutate(data):data['data'][0]['data']=bad
    # Infinity cannot be represented by HTTP JSON, inject its malformed JSON at
    # the transport boundary rather than permitting a nonstandard number.
    if bad and bad[0]==float('inf'):
        transport=httpx.MockTransport(lambda r:httpx.Response(200,content=b'{broken'))
    else:transport=native_transport('sparse',[],response_mutate=mutate)
    client=NativeRetrievalModels(config(tmp_path),None,transport)
    try:
        with pytest.raises(ModelError):client.sparse('query')
        assert not client.probe_sparse()['ready']
    finally:client.close()


@pytest.mark.parametrize('channel,status',[('dense',429),('sparse',503),('reranker',500)])
def test_http_failure_does_not_switch_model(tmp_path,channel,status):
    client=NativeRetrievalModels(config(tmp_path),None,httpx.MockTransport(lambda r:httpx.Response(status)))
    try:
        with pytest.raises(ModelError) as exc:client.request(channel,'POST','/inference',{})
        assert exc.value.retryable and str(status) in exc.value.code
    finally:client.close()


def test_reranker_native_score_is_sorted_without_changing_semantics(tmp_path):
    calls=[];client=NativeRetrievalModels(config(tmp_path),None,native_transport('reranker',calls))
    try:
        rows=client.rerank('query',['yes','no']);assert rows==[(0,.9),(1,.7)]
        b=[b for p,b in calls if p=='/score'][0];assert b['use_activation'] and b['truncation_side']=='right'
    finally:client.close()


def test_versioned_native_projection_retains_existing_canonical_and_is_host_independent(tmp_path):
    # Legacy BGE semantic identity is retained even as Qwen vector math changes.
    old=Platform(Settings(data_dir=tmp_path,retrieval_backend='legacy-http',embedding_model='BAAI/bge-m3',sparse_url='',allow_retrieval_degraded=True))
    w=old.store.create_workspace('a'*32,'legacy')['id'];scope=old.store.scope('a'*32,w,'textual');old.state.ensure_scope(scope)
    canonical=old.profile.hash;manifest=old.state._rows('SELECT * FROM manifest');old.close()
    new=Platform(config(tmp_path));profile=new.vector_profile.hash
    try:
        assert new.profile.hash==canonical and new.vector_profile.embedding_model=='Qwen/Qwen3-Embedding-0.6B'
        assert '_vllm_' in new.vector_collection and new.vector_profile.hash!=canonical
        current={r['key']:r for r in new.state._rows('SELECT * FROM manifest')}
        assert all(current[r['key']]==r for r in manifest)
    finally:new.close()
    remote=Platform(config(tmp_path,embedding_url='https://gpu.example/v1',sparse_url='https://other.example',dense_timeout=12,dense_concurrency=1))
    try:assert remote.vector_profile.hash==profile
    finally:remote.close()


def test_native_timeout_is_retryable_and_never_loads_weights(tmp_path):
    def timeout(req):raise httpx.ReadTimeout('injected',request=req)
    client=NativeRetrievalModels(config(tmp_path),None,httpx.MockTransport(timeout))
    try:
        with pytest.raises(ModelError) as exc:client.request('dense','POST','/v1/embeddings',{})
        assert exc.value.retryable
        from pathlib import Path
        source=Path(__import__('mandol_competition.chinasoft.native_retrieval',fromlist=['a']).__file__).read_text()
        assert 'from_pretrained' not in source and 'import torch' not in source
    finally:client.close()


def manager_module():
    from pathlib import Path
    import importlib.util
    spec=importlib.util.spec_from_file_location('gpu_manager_test',Path(__file__).resolve().parents[2]/'scripts/chinasoft_vllm.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_gpu_shortage_is_diagnosed_before_spawning_or_stopping(monkeypatch):
    import sys
    manager=manager_module();monkeypatch.setattr(manager,'owned',lambda c:None)
    monkeypatch.setattr(manager,'gpu_info',lambda:[{'index':'0','uuid':'test-uuid','total_mib':24000,'free_mib':100,'used_mib':23900}])
    monkeypatch.setattr(manager,'deployment',lambda c:{'managed':True,'gpu':'0','gpu_memory':.12,'python':sys.executable})
    monkeypatch.setattr(manager.subprocess,'Popen',lambda *a,**k:pytest.fail('must not spawn on shortage'))
    monkeypatch.setattr(manager.os,'killpg',lambda *a:pytest.fail('must not stop any service'))
    with pytest.raises(RuntimeError,match='No service was stopped'):manager.preflight(['dense'])


def test_remote_endpoint_placement_does_not_start_local_weights(monkeypatch):
    manager=manager_module();monkeypatch.setattr(manager,'deployment',lambda c:{'managed':False,'endpoint':'https://gpu.example'})
    monkeypatch.setattr(manager.subprocess,'Popen',lambda *a,**k:pytest.fail('remote endpoint must not spawn locally'))
    assert manager.start_one('dense')=={'channel':'dense','managed':False}


def test_lifecycle_does_not_accept_unowned_process():
    import os
    manager=manager_module()
    assert not manager.alive({'pid':os.getpid(),'boot':'wrong','ticks':'0','marker':'anything'})


def test_startup_waits_for_verified_owner_before_readiness(tmp_path,monkeypatch):
    """Real harmless child; model responses explicitly injected for this unit test."""
    import socket,sys
    import mandol_competition.chinasoft.native_retrieval as native
    manager=manager_module();monkeypatch.setattr(manager,'RUN',tmp_path)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    monkeypatch.setattr(manager,'deployment',lambda channel:dict(managed=True,endpoint=f'http://127.0.0.1:{port}',gpu='0',cache=str(tmp_path)))
    monkeypatch.setattr(manager,'verify_artifact',lambda *a:{'explicit_unit_fixture':True})
    monkeypatch.setattr(manager,'command',lambda *a:([sys.executable,'-c','import time; time.sleep(15)','vllm.entrypoints.cli.main'],None))
    real_owned=manager.owned;calls=[]
    def delayed(channel):
        value=real_owned(channel)
        if value and len(calls)<2:
            calls.append('temporarily-unverified');return None
        return value
    monkeypatch.setattr(manager,'owned',delayed)
    class InjectedClient:
        def __init__(self,*a):pass
        def probe(self,*a,**k):
            assert len(calls)==2 and real_owned('dense')
            calls.append('probe-after-ownership');return {'identity_ready':True}
        def close(self):pass
    monkeypatch.setattr(native,'NativeRetrievalModels',InjectedClient)
    def injected_warm(*a):assert real_owned('dense')
    monkeypatch.setattr(manager,'warm',injected_warm)
    try:
        result=manager.start_one('dense')
        assert result['ready'] and calls==['temporarily-unverified','temporarily-unverified','probe-after-ownership']
    finally:manager.stop_one('dense')


def test_native_credentials_use_headers_and_are_not_public(tmp_path,monkeypatch):
    monkeypatch.setenv('UNIT_VLLM_TOKEN','explicit-unit-fixture-not-a-real-secret')
    def handler(request):
        assert request.headers['authorization']=='Bearer explicit-unit-fixture-not-a-real-secret'
        assert b'explicit-unit-fixture' not in request.content
        return httpx.Response(200,json={})
    s=config(tmp_path);s.embedding_api_key_env='UNIT_VLLM_TOKEN'
    c=NativeRetrievalModels(s,None,httpx.MockTransport(handler))
    try:c.request('dense','POST','/test',{});assert 'explicit-unit-fixture' not in json.dumps(s.public())
    finally:c.close()
