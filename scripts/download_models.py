#!/usr/bin/env python3
"""Explicit public model download, pinned revisions and content-hash verification."""
from pathlib import Path
import argparse,os,sys
ROOT=Path(__file__).resolve().parents[1]
# The model environment does not install the Web application. Use only the
# bundled, stdlib-only canonical recipe helpers from this same repository.
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'packages/substrate/src')]
from mandol_competition.chinasoft.retrieval_contract import MODELS,verify_artifact

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--channel',choices=list(MODELS));args=parser.parse_args()
    cache=Path(os.getenv('CHINASOFT_MODEL_CACHE',str(ROOT/'data/models/hub'))).resolve()
    # These pinned models are acquired from their official Hub source. Do not
    # send an explicit access token to an inherited third-party mirror.
    os.environ['HF_ENDPOINT']='https://huggingface.co'
    os.environ.setdefault('HF_HOME',str(cache.parent))
    os.environ.setdefault('HF_XET_CACHE',str(cache.parent/'xet'))
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError
    # Only explicit process credentials are accepted. Never silently read a
    # developer's global HF login or another project's environment file.
    token=os.environ.get('HF_TOKEN') or False
    failures=[]
    for channel in ([args.channel] if args.channel else MODELS):
        m=MODELS[channel]
        if channel=='sparse':print('SPLADE-v3 weights: CC-BY-NC-SA-4.0. Verify your intended use before downloading.')
        try:
            snapshot_download(repo_id=m['model'],revision=m['revision'],cache_dir=str(cache),token=token,max_workers=2,allow_patterns=[m['weights'],'config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.txt','merges.txt','added_tokens.json','README.md','LICENSE'])
            verify_artifact(cache,channel);print(channel,'pinned model verified',flush=True)
        except GatedRepoError as error:
            failures.append(channel)
            from urllib.parse import urlsplit
            response=getattr(error,'response',None)
            if response is not None:
                print(f'{channel}: HTTP {response.status_code} for {urlsplit(str(response.request.url)).path}',file=sys.stderr)
            print(f'{channel}: access denied. Accept the model terms at https://huggingface.co/{m["model"]} and set an authorized HF_TOKEN; no credential is stored by this script.',file=sys.stderr)
        except Exception as error:
            failures.append(channel)
            print(f'{channel}: download/verification failed ({type(error).__name__}); check network, disk space and pinned artifact integrity. Credentials and signed download URLs are not logged.',file=sys.stderr)
    if failures:raise SystemExit('Models not ready: '+', '.join(failures))

if __name__=='__main__':main()
