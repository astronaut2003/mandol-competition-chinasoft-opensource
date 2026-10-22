"""Read-only numerical contract endpoint inside native vLLM's HTTP server.

Does not enable vLLM development/RPC/admin APIs, load weights, or run inference.
Use --middleware chinasoft_vllm_metadata.ContractMetadata and PYTHONPATH=services.
"""
import json
import os
import hmac
from pathlib import Path


class ContractMetadata:
    def __init__(self, app):
        self.app=app

    async def __call__(self, scope, receive, send):
        if scope['type']!='http' or scope['path']!='/mandol/metadata':
            return await self.app(scope,receive,send)
        key=os.getenv('VLLM_API_KEY')
        headers=dict(scope.get('headers',[]))
        if key and not hmac.compare_digest(headers.get(b'authorization',b''),('Bearer '+key).encode()):
            await send({'type':'http.response.start','status':401,'headers':[(b'content-type',b'application/json')]})
            await send({'type':'http.response.body','body':b'{"detail":"Unauthorized"}'})
            return
        # vLLM owns the config actually used by its engine, not a copied recipe.
        try:
            cfg=scope['app'].state.vllm_config;mc=cfg.model_config;hf=mc.hf_config
            pc=mc.pooler_config
            model={'dtype':str(mc.dtype).removeprefix('torch.'),'quantization':mc.quantization,
                   'max_model_len':mc.max_model_len,'runner_type':mc.runner_type,
                   'pooler_config':{k:getattr(pc,k,None) for k in ['seq_pooling_type','use_activation']},
                   'hf_config':{k:getattr(hf,k,'missing') for k in ['architectures','vocab_size','cls_token_id','sep_token_id']}}
            manifest=Path(os.environ['CHINASOFT_VLLM_MANIFEST_FILE'])
            record=json.loads(manifest.read_text())
            if Path(mc.model).name!=record['artifact']['revision']:raise ValueError('manifest does not describe running model')
            response={'vllm_config':{'model_config':model,'device_config':{'device_type':cfg.device_config.device_type}},
                      'artifact':record['artifact'],'recipe_hash':record['deployment']['recipe_hash'],
                      'attestation':'native-config-plus-locally-checksummed-cache; remote operator attested'}
            body=json.dumps(response).encode();status=200
        except Exception:
            body=b'{"detail":"native metadata unavailable or manifest mismatch"}';status=503
        await send({'type':'http.response.start','status':status,'headers':[(b'content-type',b'application/json'),(b'cache-control',b'no-store')]})
        await send({'type':'http.response.body','body':body})
