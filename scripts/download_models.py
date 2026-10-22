#!/usr/bin/env python3
"""Explicit public model download, pinned revisions and content-hash verification."""
from pathlib import Path
import argparse,os,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mandol_competition.chinasoft.retrieval_contract import MODELS,verify_artifact

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--channel',choices=list(MODELS));args=parser.parse_args()
    from huggingface_hub import snapshot_download
    cache=Path(os.getenv('CHINASOFT_MODEL_CACHE',str(ROOT/'data/models/hub'))).resolve()
    for channel in ([args.channel] if args.channel else MODELS):
        m=MODELS[channel]
        if channel=='sparse':print('SPLADE-v3 weights: CC-BY-NC-SA-4.0. Verify your intended use before downloading.')
        snapshot_download(repo_id=m['model'],revision=m['revision'],cache_dir=str(cache),allow_patterns=[m['weights'],'config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.txt','merges.txt','added_tokens.json','README.md','LICENSE'])
        verify_artifact(cache,channel);print(channel,'pinned model verified')

if __name__=='__main__':main()
