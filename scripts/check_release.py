#!/usr/bin/env python3
"""Audit only explicitly publishable source paths; never print matched secrets.

Use --list to produce a NUL-delimited staging list. Runtime paths are excluded
before reading. Gitleaks/history checks should be run separately before release.
"""
from pathlib import Path
import argparse,hashlib,json,os,re,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
ROOT_FILES={'README.md','README_EN.md','LICENSE','NOTICE','THIRD_PARTY_NOTICES.md','pyproject.toml','uv.lock','.gitignore','.dockerignore','.gitleaks.toml','.env.example','Dockerfile','docker-compose.yml','docker-compose.loopback.yml'}
DIRS=('src','packages/substrate','studio','scripts','services','docs','tests','examples','demo','deploy')
EXCLUDE={'__pycache__','node_modules','dist','build','.pytest_cache','.cache','.venv','.release-private','test-results','playwright-report'}
SUFFIXES={'.py','.pyi','.ts','.tsx','.js','.mjs','.css','.html','.json','.toml','.lock','.md','.txt','.sh','.yml','.yaml','.svg','.cfg','.ini','.mts','.in'}
SPECIAL={'LICENSE','NOTICE','.gitignore','README.txt','py.typed'}
REVIEWED_IMAGES={'demo/status_board.png','docs/assets/studio-conversation.png',
                 'docs/assets/memory-provenance.png','docs/assets/repository-symbols.png',
                 'docs/assets/retrieval-trace.png'}

def files():
 values=[ROOT/name for name in ROOT_FILES if (ROOT/name).is_file()]
 for directory in DIRS:
  base=ROOT/directory
  if not base.exists():continue
  for parent,subdirs,names in os.walk(base,followlinks=False):
   subdirs[:]=[d for d in subdirs if d not in EXCLUDE and not d.endswith('.egg-info') and not d.startswith('.venv')]
   for name in names:
    p=Path(parent)/name
    if p.name.endswith(('.pyc','.pyo','.tsbuildinfo')):continue
    values.append(p)
 return sorted(set(values))

def audit(values):
 findings=[];manifest=[]
 patterns={'private_absolute_path':re.compile(rb'/(?:home|root)/[a-zA-Z0-9_./-]+'),
           'private_key':re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----'),
           'github_pat':re.compile(rb'(?:ghp_|github_pat_)[A-Za-z0-9_]{30,}'),
           'aws_key':re.compile(rb'AKIA[0-9A-Z]{16}')}
 # Exact matching of a supplied operator credential catches any accidental
 # inclusion, without disclosing it in the report or public configuration.
 secrets=[v.encode() for k,v in os.environ.items() if (k.endswith('_API_KEY') or k.endswith('_TOKEN')) and len(v)>=20]
 for p in values:
  name=p.relative_to(ROOT).as_posix()
  if p.is_symlink():findings.append({'file':name,'rule':'symlink'});continue
  data=p.read_bytes();allowed_image=name in REVIEWED_IMAGES
  if p.suffix not in SUFFIXES and p.name not in SPECIAL and name not in ROOT_FILES and not allowed_image:findings.append({'file':name,'rule':'unreviewed_file_type'})
  if p.stat().st_size>2*1024*1024:findings.append({'file':name,'rule':'unexpected_large_file'})
  if not allowed_image:
   try:data.decode('utf8')
   except UnicodeDecodeError:findings.append({'file':name,'rule':'unreviewed_binary'})
   for rule,pattern in patterns.items():
    if pattern.search(data):findings.append({'file':name,'rule':rule})
  if any(secret in data for secret in secrets):findings.append({'file':name,'rule':'operator_credential'})
  manifest.append({'path':name,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)})
 return {'status':'PASS' if not findings else 'FAIL','file_count':len(manifest),'bytes':sum(r['bytes'] for r in manifest),'findings':findings,'files':manifest}

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--list',action='store_true');parser.add_argument('--output',type=Path);a=parser.parse_args();values=files()
 result=audit(values)
 if a.output:
  dest=a.output.resolve()
  if ROOT/'.release-private' not in dest.parents:raise SystemExit('Audit output must be in the ignored .release-private directory')
  dest.parent.mkdir(exist_ok=True,parents=True);dest.write_text(json.dumps(result,indent=2)+'\n')
 if a.list:
  if result['findings']:raise SystemExit('Audit failed; run without --list for file/rule diagnostics')
  sys.stdout.buffer.write(b''.join(p.relative_to(ROOT).as_posix().encode()+b'\0' for p in values))
 else:print(json.dumps({k:v for k,v in result.items() if k!='files'},indent=2))
 if result['findings']:raise SystemExit(1)
if __name__=='__main__':main()
