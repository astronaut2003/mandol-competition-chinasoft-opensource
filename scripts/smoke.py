#!/usr/bin/env python3
"""Real basic health check; explicit --write creates only your synthetic examples."""
from pathlib import Path
import argparse,json,os,secrets,stat,time,uuid
import httpx

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--write',action='store_true');parser.add_argument('--base-url',default='http://127.0.0.1:'+os.getenv('CHINASOFT_PORT','20600'));parser.add_argument('--username',default=os.getenv('MANDOL_SMOKE_USERNAME'));parser.add_argument('--password-file',type=Path,default=os.getenv('MANDOL_SMOKE_PASSWORD_FILE'));a=parser.parse_args()
    report={'kind':'real-http-smoke','checks':[]}
    with httpx.Client(base_url=a.base_url,trust_env=False,timeout=45,headers={'X-Mandol-Client':'studio'}) as c:
        for path in ['/healthz','/']:
            r=c.get(path);r.raise_for_status();report['checks'].append({'path':path,'status':r.status_code})
        if a.username:
            if not a.password_file or a.password_file.stat().st_mode&0o077:raise SystemExit('Provide a private 0600 password file; credentials are never taken from command arguments.')
            r=c.post('/api/auth/login',json={'username':a.username,'password':a.password_file.read_text().strip()});r.raise_for_status();identity=r.json();c.headers['X-CSRF-Token']=identity['csrf_token']
            w=identity['workspaces'][0]['id'];s=c.get('/api/status',params={'workspace_id':w});s.raise_for_status();status=s.json()
            report['dependencies']={'sqlite':status['sqlite'],'qdrant':status['qdrant'],'three_channel_ready':status.get('retrieval',{}).get('three_channel_ready',False),'workers_alive':all(x['alive'] for x in status['workers'])}
            if a.write:
                # Stable names reuse your own fixture workspace; request IDs are
                # new so repeated smoke does not pretend old jobs ran again.
                workspaces=c.get('/api/workspaces').json()['items'];work=next((w for w in workspaces if w['name']=='Public smoke examples'),None)
                if not work:r=c.post('/api/workspaces',json={'name':'Public smoke examples'});r.raise_for_status();work=r.json()
                w=work['id'];r=c.get('/api/sessions',params={'workspace_id':w,'q':'Synthetic smoke'});r.raise_for_status();sessions=r.json()['items']
                if sessions:session=sessions[0]
                else:r=c.post('/api/sessions',json={'workspace_id':w,'title':'Synthetic smoke','unified':True});r.raise_for_status();session=r.json()
                request_id=str(uuid.uuid4());text='Synthetic fixture: the Aurora demo workspace uses a blue notebook.'
                payload={'workspace_id':w,'session_id':session['id'],'request_id':request_id,'messages':[{'role':'user','content':text}]}
                r=c.post('/api/memories',json=payload);r.raise_for_status();receipt=r.json()
                again=c.post('/api/memories',json=payload);again.raise_for_status();assert again.json()['memory_ids']==receipt['memory_ids']
                end=time.monotonic()+45
                while time.monotonic()<end:
                    state=c.get('/api/ingestions/'+receipt['ingestion_id'],params={'workspace_id':w});state.raise_for_status()
                    if state.json()['status']=='saved':break
                    time.sleep(.3)
                else:raise SystemExit('Durable input not yet ingested; inspect Background Jobs.')
                value=c.get('/api/memories/'+receipt['memory_ids'][0],params={'workspace_id':w});value.raise_for_status();assert value.json()['text']==text
                report['write']={'canonical_saved':True,'idempotent_replay':True,'memory_id':receipt['memory_ids'][0],'workspace_id':w,'session_id':session['id']}
                jobs=c.get('/api/jobs',params={'workspace_id':w,'limit':50}).json()['items'];report['job_states']={kind:[x['status'] for x in jobs if x['kind']==kind] for kind in ['ingest','index','derive']}
            c.post('/api/auth/logout',json={}).raise_for_status()
        elif a.write:raise SystemExit('--write requires your existing username and --password-file. Create an account with ./scripts/admin.sh first.')
    report['pass']=True;print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
