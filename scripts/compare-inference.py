#!/usr/bin/env python3
"""Capture synthetic GPU outputs and compare two runtime installations.

Only calls explicitly configured inference endpoints. Never reads user data or
writes an index. Evidence files contain public synthetic fixtures and numbers.
"""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import time
import urllib.request

TEXTS = [
    'The camera is inside the blue cabinet.',
    'Bananas grow in warm weather.',
    '用户的饮品偏好已更正为无糖绿茶，旧偏好咖啡已经失效。',
    'The service persists canonical memories in SQLite and vectors in Qdrant.',
    'def verify_source(version): return version.active and not version.deleted',
    'Dense retrieval and SPLADE sparse retrieval are fused with BM25 using RRF.',
]
QUERIES = ['Where is the camera?', '用户现在喜欢喝什么？', 'How are sources verified?']
PROMPT = 'Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: '
MODELS = {'dense': 'qwen3-embedding-0.6b', 'sparse': 'splade-v3',
          'reranker': 'BAAI/bge-reranker-v2-m3'}


def rows(value, field):
    return [r[field] for r in sorted(value['data'], key=lambda r: r['index'])]


def capture(args):
    http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    report = {'fixtures': {'documents': TEXTS, 'queries': QUERIES}, 'channels': {}}
    for channel in MODELS:
        base = getattr(args, channel).rstrip('/').removesuffix('/v1')
        key_name = getattr(args, channel + '_key_env')
        key = os.environ[key_name] if key_name else None

        def call(path, body=None):
            headers = {'Content-Type': 'application/json'}
            if key:
                headers['Authorization'] = 'Bearer ' + key
            request = urllib.request.Request(base + path, headers=headers,
                data=None if body is None else json.dumps(body).encode())
            with http.open(request, timeout=120) as response:
                return json.load(response)

        r = {'version': call('/version'), 'metadata': call('/mandol/metadata')}
        if channel == 'dense':
            def infer(texts):
                return rows(call('/v1/embeddings', {'model': MODELS[channel], 'input': texts,
                    'encoding_format': 'float', 'truncate_prompt_tokens': 4096,
                    'truncation_side': 'right'}), 'embedding')
            inputs = TEXTS + [PROMPT + q for q in QUERIES]
        elif channel == 'sparse':
            def infer(texts):
                return rows(call('/pooling', {'model': MODELS[channel], 'input': texts,
                    'task': 'embed', 'encoding_format': 'float', 'add_special_tokens': True,
                    'truncate_prompt_tokens': 512, 'truncation_side': 'right',
                    'use_activation': False}), 'data')
            inputs = TEXTS + QUERIES
        else:
            def infer(texts, activation=True):
                return rows(call('/score', {'model': MODELS[channel], 'text_1': QUERIES[0],
                    'text_2': texts, 'truncate_prompt_tokens': 4096,
                    'truncation_side': 'right', 'use_activation': activation}), 'score')
            inputs = TEXTS
        r['single'] = [infer([text])[0] for text in inputs]
        # Bounded pairs use the deployment's qualified max-num-seqs=2.
        r['batch'] = []
        for i in range(0, len(inputs), 2):
            r['batch'].extend(infer(inputs[i:i + 2]))
        if channel == 'reranker':
            r['raw_logits'] = infer(TEXTS, False)
        elapsed = []
        for _ in range(args.samples):
            began = time.perf_counter()
            infer(inputs[:1] if channel != 'reranker' else inputs[:2])
            elapsed.append((time.perf_counter() - began) * 1000)
        r['single_or_pair_latency_ms'] = elapsed
        r['latency_p50_ms'] = statistics.median(elapsed)
        report['channels'][channel] = r
    return report


def cosine(a, b):
    return sum(x*y for x, y in zip(a, b, strict=True)) / (math.hypot(*a)*math.hypot(*b))


def compare(args):
    old, new = [json.loads(p.read_text()) for p in (args.baseline, args.candidate)]
    assert old['fixtures'] == new['fixtures'], 'Different fixtures cannot be compared'
    report = {'channels': {}, 'tolerances': {
        'rationale': 'BF16 inference; compare float32 HTTP outputs, including batch rounding. '
                     'Sparse zero crossings are reported separately from cosine and weighted error.',
        'dense_max_abs': .005, 'dense_min_cosine': .999,
        'sparse_max_abs': .08, 'sparse_min_cosine': .999, 'sparse_min_token_jaccard': .98,
        'reranker_max_score_abs': .02, 'reranker_max_logit_abs': .125}}
    passed = True
    for channel in MODELS:
        a, b = old['channels'][channel], new['channels'][channel]
        # Verify model bytes, not just a served alias.
        artifact_keys = ['model', 'revision', 'weights_sha256', 'config.json', 'tokenizer.json']
        identity = all(a['metadata']['artifact'][k] == b['metadata']['artifact'][k]
                       for k in artifact_keys)
        r = {'identity_equal': identity, 'old_version': a['version'], 'new_version': b['version']}
        if channel != 'reranker':
            pairs = list(zip(a['single'], b['single'], strict=True))
            r.update(dimensions=sorted({len(v) for pair in pairs for v in pair}),
                max_abs=max(abs(x-y) for va, vb in pairs for x, y in zip(va, vb, strict=True)),
                min_cosine=min(cosine(va, vb) for va, vb in pairs),
                norms_old=[math.hypot(*v) for v in a['single']],
                norms_new=[math.hypot(*v) for v in b['single']],
                new_batch_max_abs=max(abs(x-y) for va, vb in zip(b['single'], b['batch'], strict=True)
                                      for x, y in zip(va, vb, strict=True)))
            ok = r['min_cosine'] >= .999 and r['max_abs'] <= (.005 if channel == 'dense' else .08)
            if channel == 'dense':
                ok &= r['dimensions'] == [1024] and all(abs(n-1) < .02 for n in r['norms_new'])
                ok &= r['new_batch_max_abs'] < .02
            else:
                sets = [(set(i for i, x in enumerate(va) if x > 0),
                         set(i for i, x in enumerate(vb) if x > 0)) for va, vb in pairs]
                r['token_jaccard'] = [len(x & y)/len(x | y) for x, y in sets]
                r['nonzero_old'] = [len(x) for x, _ in sets]
                r['nonzero_new'] = [len(y) for _, y in sets]
                r['top_tokens_old'] = [sorted(s, key=lambda i: va[i], reverse=True)[:10]
                                      for (va, _), (s, _) in zip(pairs, sets)]
                r['top_tokens_new'] = [sorted(s, key=lambda i: vb[i], reverse=True)[:10]
                                      for (_, vb), (_, s) in zip(pairs, sets)]
                r['retrieval_orders_old'] = [sorted(range(len(TEXTS)), key=lambda i: sum(x*y for x,y in zip(q,a['single'][i])), reverse=True) for q in a['single'][len(TEXTS):]]
                r['retrieval_orders_new'] = [sorted(range(len(TEXTS)), key=lambda i: sum(x*y for x,y in zip(q,b['single'][i])), reverse=True) for q in b['single'][len(TEXTS):]]
                ok &= r['dimensions'] == [30522] and min(r['token_jaccard']) >= .98
                ok &= r['new_batch_max_abs'] < .08
                ok &= all(x[:3] == y[:3] for x,y in zip(r['retrieval_orders_old'], r['retrieval_orders_new']))
        else:
            r['max_score_abs'] = max(abs(x-y) for x,y in zip(a['single'], b['single'], strict=True))
            r['max_logit_abs'] = max(abs(x-y) for x,y in zip(a['raw_logits'], b['raw_logits'], strict=True))
            r['new_batch_max_abs'] = max(abs(x-y) for x,y in zip(b['single'], b['batch'], strict=True))
            r['order_old'] = sorted(range(len(TEXTS)), key=lambda i: a['single'][i], reverse=True)
            r['order_new'] = sorted(range(len(TEXTS)), key=lambda i: b['single'][i], reverse=True)
            ok = r['max_score_abs'] <= .02 and r['max_logit_abs'] <= .125
            ok &= r['order_old'] == r['order_new'] and r['new_batch_max_abs'] <= .02
        r['status'] = 'PASS' if identity and ok else 'FAIL'
        passed &= r['status'] == 'PASS'
        report['channels'][channel] = r
    report['status'] = 'PASS' if passed else 'FAIL'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    capture_parser = sub.add_parser('capture')
    for c in MODELS:
        capture_parser.add_argument('--' + c, required=True)
        capture_parser.add_argument('--' + c + '-key-env', default='')
    capture_parser.add_argument('--samples', type=int, default=12)
    compare_parser = sub.add_parser('compare')
    compare_parser.add_argument('--baseline', type=Path, required=True)
    compare_parser.add_argument('--candidate', type=Path, required=True)
    for p in [capture_parser, compare_parser]:
        p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = capture(args) if args.action == 'capture' else compare(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    args.output.chmod(0o600)
    print(json.dumps({'action': args.action, 'status': result.get('status', 'CAPTURED')}))
    if result.get('status') == 'FAIL':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
