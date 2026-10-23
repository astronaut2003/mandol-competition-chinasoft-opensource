#!/usr/bin/env python3
"""Independent native API + Compose Qdrant lifecycle. Never manages other projects."""
from pathlib import Path
import argparse,contextlib,fcntl,json,os,signal,socket,subprocess,sys,time,urllib.request
ROOT=Path(__file__).resolve().parents[1]
from mandol_competition.chinasoft.config import Settings
RUN=ROOT/'.runtime';RECORD=RUN/'api.json'


def health(url):
    try:
        with urllib.request.urlopen(url,timeout=2) as r:return r.status==200
    except Exception:return False


def process(record):
    try:
        pid=record['pid'];fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        args=Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
        return fields[0]!='Z' and fields[19]==record['ticks'] and record['boot']==Path('/proc/sys/kernel/random/boot_id').read_text().strip() and os.getpgid(pid)==pid and str(ROOT/'src') in args and 'mandol_competition.chinasoft.api:create_app' in args
    except (OSError,KeyError):return False


def owned():
    try:r=json.loads(RECORD.read_text())
    except (OSError,ValueError):return None
    return r if process(r) else None


def compose(*args):
    project=os.getenv('MANDOL_COMPOSE_PROJECT','mandol-os-native')
    if not project.startswith('mandol-os-'):raise RuntimeError('Compose project must start with mandol-os-')
    return ['docker','compose','--project-directory',str(ROOT),'-f',str(ROOT/'docker-compose.yml'),'-p',project,*args]


def port_free(port):
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        try:sock.bind(('127.0.0.1',port));return True
        except OSError:return False


def guard(s):
    # Reserve the known local competition ranges. These are guards, never targets.
    reserved={18600,18601,18602,18610,18633,19600,19631,19632,19633,19635,19636}
    from urllib.parse import urlsplit
    if s.port in reserved:raise RuntimeError('Choose an independent application port (default 20600).')
    for endpoint in [s.qdrant_url,s.embedding_url,s.sparse_url,s.reranker_url]:
        u=urlsplit(endpoint)
        if u.hostname in ('127.0.0.1','localhost') and u.port in reserved:
            raise RuntimeError('A configured endpoint targets a reserved local deployment. Use independent services/ports.')


def status(s):
    r=owned();url='http://127.0.0.1:'+str(r['port'] if r else s.port)
    value={'url':url,'api_owned':bool(r),'api_health':health(url+'/healthz'),'qdrant_health':health(s.qdrant_url+'/readyz'),'data_dir':str(s.data_dir),'retrieval_mode':r.get('mode') if r else None,'gpu_started_by_application':False}
    print(json.dumps(value,ensure_ascii=False,indent=2));return value


def init(s):
    import sqlite3
    current=owned()
    if current:
        if current['data_dir']!=str(s.data_dir):raise RuntimeError('Owned API uses a different data directory; stop it before initializing another database.')
        with sqlite3.connect((s.data_dir/'state.sqlite3').as_uri()+'?mode=ro',uri=True) as db:
            check=db.execute('PRAGMA quick_check').fetchone()[0]
        if check!='ok':raise RuntimeError('SQLite quick_check failed')
        print('Owned API is running; SQLite read-only check passed. Schema initialization deferred to the next normal restart.')
        return
    from mandol_competition.chinasoft.service import Platform
    try:p=Platform(s)
    except BlockingIOError as exc:raise RuntimeError('SQLite is owned by another process. No migration applied; stop its operator-managed instance before setup.') from exc
    try:
        import sqlite3
        with sqlite3.connect(s.data_dir/'state.sqlite3') as db:
            check=db.execute('PRAGMA quick_check').fetchone()[0]
        if check != 'ok':raise RuntimeError('SQLite quick_check failed')
        print('SQLite schema initialized; no account/model/service started:',check)
    finally:p.close()


def start(s,basic):
    guard(s)
    if not (ROOT/'studio/dist/index.html').is_file():raise RuntimeError('Build frontend first: ./scripts/setup.sh')
    r=owned()
    if r:
        if r['port']!=s.port or r['data_dir']!=str(s.data_dir):raise RuntimeError('Existing owned API uses different configuration. Stop it with its original configuration first.')
        print('Owned API already running; preserving current data and process.');return status(s)
    if not port_free(s.port):raise RuntimeError('Application port is occupied by an unowned process; no process was changed.')
    if not basic and not s.allow_retrieval_degraded:
        from mandol_competition.chinasoft.native_retrieval import NativeRetrievalModels
        c=NativeRetrievalModels(s,None)
        try:ready=c.probe_all(force=True)
        finally:c.close()
        missing=[key for key,value in ready.items() if not value['ready']]
        if missing:raise RuntimeError('Required retrieval services unavailable/incompatible: '+', '.join(missing)+'. Configure independent HTTP endpoints or explicitly provision local GPU services. For a basic storage/UI check use --basic; this does not claim full retrieval readiness.')
    qport=int(os.getenv('MANDOL_QDRANT_PORT','20633'))
    local=s.qdrant_url.rstrip('/')==f'http://127.0.0.1:{qport}'
    if not health(s.qdrant_url+'/readyz'):
        if not local:raise RuntimeError('Configured independent Qdrant is not ready.')
        if not port_free(qport):raise RuntimeError('Qdrant port is occupied by an unowned service.')
        subprocess.run(compose('up','-d','qdrant'),cwd=ROOT,check=True)
        for _ in range(90):
            if health(s.qdrant_url+'/readyz'):break
            time.sleep(.5)
        else:raise RuntimeError('Qdrant did not become ready. Inspect your own Compose logs.')
    elif local:
        ps=subprocess.check_output(compose('ps','--format','json','qdrant'),cwd=ROOT,text=True)
        if not ps.strip():raise RuntimeError('An unowned Qdrant listener occupies the configured port; refusing to adopt it.')
    environment=dict(os.environ)
    if basic:environment['CHINASOFT_ALLOW_RETRIEVAL_DEGRADED']='true'
    with (RUN/'api.log').open('ab') as log:
        p=subprocess.Popen([sys.executable,'-m','uvicorn','mandol_competition.chinasoft.api:create_app','--factory','--app-dir',str(ROOT/'src'),'--host',s.host,'--port',str(s.port),'--workers','1','--no-access-log','--timeout-graceful-shutdown','90'],cwd=ROOT,env=environment,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    r={'pid':p.pid,'ticks':Path(f'/proc/{p.pid}/stat').read_text().rsplit(')',1)[1].split()[19],'boot':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'port':s.port,'data_dir':str(s.data_dir),'mode':'explicit-basic' if basic or s.allow_retrieval_degraded else 'full'}
    RECORD.write_text(json.dumps(r,indent=2)+'\n');os.chmod(RECORD,0o600)
    for _ in range(120):
        if health(f'http://127.0.0.1:{s.port}/healthz'):return status(s)
        if p.poll() is not None:raise RuntimeError('API exited during startup; see .runtime/api.log. Models were not replaced or mocked.')
        time.sleep(.5)
    raise RuntimeError('API did not become healthy; see .runtime/api.log. The owned process is retained for diagnosis.')


def stop(s,api_only=False):
    r=owned()
    if r:
        os.killpg(r['pid'],signal.SIGTERM);deadline=time.monotonic()+120
        while process(r) and time.monotonic()<deadline:time.sleep(.25)
        if process(r):raise RuntimeError('API is still draining. No force-kill sent; check the owned log.')
        RECORD.unlink(missing_ok=True)
    if api_only:
        print('Stopped only the owned API; Qdrant, model services and persistent data were retained.')
        return
    subprocess.run([sys.executable,str(ROOT/'scripts/chinasoft_vllm.py'),'stop'],check=True,cwd=ROOT)
    subprocess.run(compose('down','--timeout','45'),check=True,cwd=ROOT)
    print('Stopped only this project. Persistent volumes, SQLite and uploads were retained.')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['init','start','status','stop']);parser.add_argument('--basic',action='store_true');parser.add_argument('--api-only',action='store_true',help='Stop only the owned API, keeping Qdrant and GPU services');a=parser.parse_args()
    if a.api_only and a.action!='stop':parser.error('--api-only is only available for stop')
    RUN.mkdir(exist_ok=True,mode=0o700);s=Settings()
    with (RUN/'lifecycle.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if a.action=='init':init(s)
        elif a.action=='start':start(s,a.basic)
        elif a.action=='stop':stop(s,a.api_only)
        else:status(s)

if __name__=='__main__':
    try:main()
    except (RuntimeError,ValueError,subprocess.SubprocessError) as e:print(str(e),file=sys.stderr);sys.exit(1)
