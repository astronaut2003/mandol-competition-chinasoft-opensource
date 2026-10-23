"""Immutable numerical recipes, independent of hosts, GPU numbers and secrets.

Reuses the already qualified Mandol native vLLM recipes. No model imports.
Changes to precision, truncation, model or pooling require a fresh projection.
"""
from pathlib import Path
import hashlib
import json

MODELS = {
    'dense': {'model': 'Qwen/Qwen3-Embedding-0.6B', 'revision': '97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3',
              'alias': 'qwen3-embedding-0.6b', 'version': '0.32.0', 'max_length': 4096,
              'pooler': {'seq_pooling_type': 'LAST', 'use_activation': True}, 'port': 20631,
              'gpu_memory': .12, 'weights': 'model.safetensors',
              'weights_sha256':'0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd'},
    'sparse': {'model': 'naver/splade-v3', 'revision': 'fdfeceb91d7b9de7985b38addd3ba9f53a59a355',
               'alias': 'splade-v3', 'version': '0.32.0', 'max_length': 512,
               'pooler': {'use_activation': False}, 'port': 20636, 'gpu_memory': .10,
               'overrides': {'architectures': ['BertSpladeSparseEmbeddingModel'], 'cls_token_id': None, 'sep_token_id': None},
               'weights': 'pytorch_model.bin',
               'weights_sha256':'e59283d1f2db78da12abc9dc9ffd499f4a58c66652ad746e4d9bd01adc383077'},
    'reranker': {'model': 'BAAI/bge-reranker-v2-m3', 'revision': '953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e',
                 'alias': 'BAAI/bge-reranker-v2-m3', 'version': '0.32.0', 'max_length': 4096,
                 'pooler': {}, 'port': 20632, 'gpu_memory': .14, 'weights': 'model.safetensors',
                 'weights_sha256':'d9e3e081faff1eefb84019509b2f5558fd74c1a05a2c7db22f74174fcedb5286'},
}
QUERY_PROMPT = 'Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: '


def recipe(channel):
    m = MODELS[channel]
    return {'schema': 'chinasoft-native-vllm-gpu-v1', 'channel': channel, 'model': m['model'],
            'revision': m['revision'], 'weights_sha256':m['weights_sha256'], 'vllm_version': m['version'], 'dtype': 'bfloat16',
            'quantization': 'none', 'max_length': m['max_length'], 'truncation': 'right',
            'input_char_prefix': 4000, 'pooler': m['pooler'], 'hf_overrides': m.get('overrides', {}),
            'query_prompt': QUERY_PROMPT if channel == 'dense' else '',
            'pooling': {'dense': 'last-token-l2-normalized', 'sparse': 'all-token-max-log1p-relu',
                        'reranker': 'cls-cross-encoder-sigmoid'}[channel]}


def recipe_hash(channel):
    # Same canonical serialization as the substrate. Keep model download and
    # reference tools independent of the application and its dependencies.
    value=json.dumps(recipe(channel),sort_keys=True,separators=(',', ':'),ensure_ascii=False,allow_nan=False)
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def cache_snapshot(cache, channel):
    m = MODELS[channel]
    return Path(cache)/('models--' + m['model'].replace('/', '--'))/'snapshots'/m['revision']


def verify_artifact(cache, channel):
    """Verify actual weight bytes; Hub cache names may be opaque Xet IDs."""
    path = cache_snapshot(cache, channel)
    m = MODELS[channel]
    weight = path/m['weights']
    with weight.open('rb') as f:
        sha = hashlib.file_digest(f, 'sha256').hexdigest()
    if sha != m['weights_sha256']:
        raise ValueError('Cached weight bytes differ from the pinned SHA256: ' + channel)
    names = ['config.json', 'tokenizer.json', 'tokenizer_config.json']
    files = {n: hashlib.sha256((path/n).read_bytes()).hexdigest() for n in names}
    return {'model': m['model'], 'revision': m['revision'], 'weights_sha256': sha, **files}
