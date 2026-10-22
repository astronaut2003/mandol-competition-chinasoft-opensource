#!/usr/bin/env python3
"""Owned, independent native vLLM GPU services; never adopts or kills outsiders."""
from pathlib import Path
import argparse, fcntl, hashlib, json, os, signal, socket, subprocess, sys, time
from dataclasses import replace
from urllib.parse import urlsplit
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mandol_competition.chinasoft.config import Settings,env
from mandol_competition.chinasoft.retrieval_contract import MODELS,cache_snapshot,verify_artifact,recipe,recipe_hash
from mandol_competition.chinasoft.models import ModelError
RUN=ROOT/'.runtime/vllm-gpu'


def settings():return Settings()


def deployment(channel):
    s=settings();m=MODELS[channel];prefix=channel.upper()
    url={'dense':s.embedding_url,'sparse':s.sparse_url,'reranker':s.reranker_url}[channel].rstrip('/').removesuffix('/v1')
    default='http://127.0.0.1:'+str(m['port'])
    manage=env(prefix+'_MANAGED','false')
    if manage not in ('auto','true','false'):raise ValueError('MANAGED must be auto/true/false')
    managed=url==default if manage=='auto' else manage=='true'
    device=env(prefix+'_GPU','0');fraction=float(env(prefix+'_GPU_MEMORY',m['gpu_memory']))
    seqs=int(env(prefix+'_MAX_SEQS',2));batched=int(env(prefix+'_MAX_BATCHED_TOKENS',m['max_length']))
    if not 0<fraction<=.5 or not 1<=seqs<=8 or not m['max_length']<=batched<=m['max_length']*4:
        raise ValueError('Invalid bounded GPU memory/sequence/token budget')
    parsed=urlsplit(url)
    if managed and (parsed.hostname not in ('127.0.0.1','localhost') or not parsed.port or parsed.path):
        raise ValueError('Managed services require an explicit loopback origin; use MANAGED=false for remote endpoints')
    version='025' if channel=='reranker' else '029'
    python=Path(env(prefix+'_VLLM_PYTHON',ROOT/('.venv-vllm-'+version)/'bin/python'))
    cache=Path(env('MODEL_CACHE',ROOT/'data/models/hub')).expanduser().resolve()
    return {'managed':managed,'endpoint':url,'gpu':device,'gpu_memory':fraction,'max_seqs':seqs,
            'max_batched_tokens':batched,'python':str(python),'cache':str(cache),'recipe_hash':recipe_hash(channel)}


def alive(record):
    try:
        pid=record['pid'];stat=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
        return (record['boot']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
                and record['ticks']==stat[19] and stat[0]!='Z'
                and record['marker'] in cmd and str(ROOT) in cmd and os.getpgid(pid)==pid)
    except (OSError,KeyError):return False


def owned(channel):
    file=RUN/(channel+'.json')
    record=json.loads(file.read_text()) if file.exists() else None
    return record if record and alive(record) else None


def gpu_info():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.free,memory.used',
        '--format=csv,noheader,nounits'],text=True,timeout=8)
    result=[]
    for line in raw.strip().splitlines():
        i,uuid,name,total,free,used=[v.strip() for v in line.split(',')]
        result.append({'index':i,'uuid':uuid,'name':name,'total_mib':int(total),'free_mib':int(free),'used_mib':int(used)})
    return result


def preflight(channels):
    gpus=gpu_info();demands={}
    for channel in channels:
        d=deployment(channel)
        if not d['managed'] or owned(channel):continue
        found=[g for g in gpus if d['gpu'] in (g['index'],g['uuid'])]
        if len(found)!=1:raise RuntimeError('GPU device unavailable for '+channel+'; set CHINASOFT_'+channel.upper()+'_GPU to a device index or UUID')
        g=found[0];demands[g['index']]=demands.get(g['index'],0)+d['gpu_memory']*g['total_mib']
        if not Path(d['python']).is_file():raise RuntimeError('vLLM environment unavailable: run scripts/inference-setup.sh or configure per-channel VLLM_PYTHON')
    for g in gpus:
        required=demands.get(g['index'],0)
        if required and required+2048>g['free_mib']:
            raise RuntimeError(f"GPU {g['index']}: free {g['free_mib']} MiB, declared new budgets {required:.0f} MiB + 2048 MiB reserve. No service was stopped. Choose another GPU/remote endpoint or lower bounded GPU_MEMORY/MAX_SEQS budgets; do not terminate unrelated services.")
    # Resource diagnostics precede downloads; all selected artifacts must still
    # pass before allocating any new GPU process.
    for channel in channels:
        d=deployment(channel)
        if d['managed'] and not owned(channel):verify_artifact(d['cache'],channel)
    return gpus


def command(channel,d):
    m=MODELS[channel];snapshot=cache_snapshot(d['cache'],channel)
    result=[d['python'],'-m','vllm.entrypoints.cli.main','serve',str(snapshot),'--runner','pooling','--dtype','bfloat16',
        '--enforce-eager','--max-model-len',str(m['max_length']),'--max-num-seqs',str(d['max_seqs']),
        '--max-num-batched-tokens',str(d['max_batched_tokens']),'--gpu-memory-utilization',str(d['gpu_memory']),
        '--served-model-name',m['alias'],'--host','127.0.0.1','--port',str(urlsplit(d['endpoint']).port),
        '--disable-uvicorn-access-log','--middleware','chinasoft_vllm_metadata.ContractMetadata']
    if m['pooler']:result+=['--pooler-config',json.dumps(m['pooler'])]
    if m.get('overrides'):result+=['--hf-overrides',json.dumps(m['overrides'])]
    key_env={'dense':settings().embedding_api_key_env,'sparse':settings().sparse_api_key_env,'reranker':settings().reranker_api_key_env}[channel]
    if key_env:
        # Never place API credentials in argv or lifecycle records.
        if not os.getenv(key_env):raise ValueError('Configured native service API key environment is missing')
    return result,key_env


def start_one(channel):
    d=deployment(channel)
    if not d['managed']:return {'channel':channel,'managed':False}
    existing=owned(channel)
    if existing:
        if existing['deployment']!=d:raise RuntimeError('Owned '+channel+' has a different deployment; explicitly stop it before changing resources')
        from mandol_competition.chinasoft.native_retrieval import NativeRetrievalModels
        client=NativeRetrievalModels(replace(settings(),dense_timeout=120,sparse_timeout=120,reranker_timeout=120),None)
        try:
            if not client.probe(channel,force=True).get('identity_ready'):raise RuntimeError('Owned native service has incompatible metadata: '+channel)
            warm(client,channel)
            return {'channel':channel,'already_running':True,'pid':existing['pid'],'ready':True}
        finally:client.close()
    port=urlsplit(d['endpoint']).port
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        try:sock.bind(('127.0.0.1',port))
        except OSError:raise RuntimeError('Unowned listener on port '+str(port)+'; refusing to stop or adopt it') from None
    artifact=verify_artifact(d['cache'],channel)
    cmd,key_env=command(channel,d)
    environment=dict(os.environ)
    environment.update(CUDA_VISIBLE_DEVICES=d['gpu'],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        VLLM_NO_USAGE_STATS='1',VLLM_USE_V2_MODEL_RUNNER='0',VLLM_SERVER_DEV_MODE='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='2',VLLM_RPC_BASE_PATH='/tmp',VLLM_CACHE_ROOT=str(RUN/'cache'/channel),
        HF_HUB_CACHE=d['cache'],PYTHONUNBUFFERED='1')
    environment['PYTHONPATH']=str(ROOT/'services')
    environment['CHINASOFT_VLLM_MANIFEST_FILE']=str(RUN/(channel+'.json'))
    if key_env:environment['VLLM_API_KEY']=os.getenv(key_env)
    else:environment.pop('VLLM_API_KEY',None)
    with (RUN/(channel+'.log')).open('ab') as log:
        process=subprocess.Popen(cmd,cwd=ROOT,env=environment,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    record={'pid':process.pid,'ticks':Path(f'/proc/{process.pid}/stat').read_text().rsplit(')',1)[1].split()[19],
        'boot':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'marker':'vllm.entrypoints.cli.main',
        'deployment':d,'recipe':recipe(channel),'artifact':artifact,'command':cmd,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    (RUN/(channel+'.json')).write_text(json.dumps(record,indent=2))
    from mandol_competition.chinasoft.native_retrieval import NativeRetrievalModels
    client=NativeRetrievalModels(replace(settings(),dense_timeout=120,sparse_timeout=120,reranker_timeout=120),None);deadline=time.monotonic()+300
    last_reason=None;ownership_deadline=time.monotonic()+5
    try:
        while time.monotonic()<deadline:
            if not owned(channel):
                # /proc can transiently expose an empty/changing command line
                # during exec. Only the Popen we just created gets a short
                # grace window. No probe/adoption/readiness is allowed before
                # the full boot/ticks/marker/process-group checks succeed.
                if process.poll() is not None or time.monotonic()>=ownership_deadline:
                    raise RuntimeError(channel+' exited or startup ownership is unverified; inspect .runtime/vllm-gpu/'+channel+'.log for CUDA/vLLM diagnostics. No CPU/model fallback.')
                time.sleep(.1);continue
            health=client.probe(channel,force=True)
            if health.get('identity_ready'):
                # Readiness includes real inference, not only a listening socket.
                warm(client,channel)
                return {'channel':channel,'pid':process.pid,'ready':True,'recipe_hash':recipe_hash(channel)}
            if health['reason']!=last_reason:
                print('Waiting for '+channel+': '+health['reason'],flush=True);last_reason=health['reason']
            time.sleep(1)
        raise RuntimeError(channel+' contract/readiness timeout; inspect owned log. No unowned service changed.')
    finally:client.close()


def warm(client,channel):
    if channel=='dense':client.dense(['ChinaSoft retrieval readiness'],mode='query')
    elif channel=='sparse':client.sparse_batch(['ChinaSoft retrieval readiness'],mode='query')
    else:
        scores=client.rerank('Where is the camera?',['The camera is in the cabinet.','Bananas are yellow.'])
        if dict(scores)[0]<=dict(scores)[1]:raise RuntimeError('BGE relevance orientation probe failed')


def stop_one(channel):
    r=owned(channel)
    if not r:return
    os.killpg(r['pid'],signal.SIGTERM);deadline=time.monotonic()+45
    while owned(channel) and time.monotonic()<deadline:time.sleep(.2)
    if owned(channel):raise RuntimeError('Owned service did not stop gracefully: '+channel+'; no unverified kill was sent')
    (RUN/(channel+'.json')).unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['start','stop','status','diagnose']);parser.add_argument('--channel',choices=list(MODELS));args=parser.parse_args()
    RUN.mkdir(parents=True,exist_ok=True)
    with (RUN/'.lifecycle.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        channels=[args.channel] if args.channel else list(MODELS)
        if args.action=='start':
            before=preflight(channels);result=[]
            created=[]
            try:
                for c in channels:
                    if deployment(c)['managed'] and not owned(c):created.append(c)
                    began=time.monotonic();r=start_one(c);r['seconds']=round(time.monotonic()-began,3);result.append(r)
                    print(json.dumps(r),flush=True)
            except Exception:
                # Roll back only this attempt's new processes. Previously running
                # owned services and every unowned listener remain untouched.
                for c in reversed(created):
                    try:stop_one(c)
                    except RuntimeError:print('Cleanup could not verify/stop '+c+'; inspect its owned log.',file=sys.stderr)
                raise
            (RUN/'startup.json').write_text(json.dumps({'services':result,'gpu_before':before,'gpu_after':gpu_info()},indent=2))
        elif args.action=='stop':
            for c in reversed(channels):stop_one(c)
            print('Stopped only manifest-verified ChinaSoft vLLM process groups')
        elif args.action=='diagnose':print(json.dumps({'gpus':gpu_info(),'services':{c:deployment(c) for c in channels}},indent=2))
        else:
            from mandol_competition.chinasoft.native_retrieval import NativeRetrievalModels
            client=NativeRetrievalModels(settings(),None)
            try:print(json.dumps({'health':client.probe_all(force=True),'owned':{c:(owned(c) or {}).get('pid') for c in channels}},ensure_ascii=False,indent=2))
            finally:client.close()


if __name__=='__main__':
    try:main()
    except (ValueError,RuntimeError,FileNotFoundError,subprocess.SubprocessError,ModelError) as exc:
        print(str(exc),file=sys.stderr);sys.exit(1)
