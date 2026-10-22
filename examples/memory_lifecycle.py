#!/usr/bin/env python3
"""Create synthetic sources and inspect real version/provenance APIs.

Run with the project .venv. No fixed passwords, external keys or private data.
High-level output requires configured models; --wait-higher is an optional wait,
not a promise that a model will extract a specific fact.
"""
import argparse,json,time,uuid
from pathlib import Path
import httpx

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base-url',default='http://127.0.0.1:20600');p.add_argument('--username',required=True);p.add_argument('--password-file',type=Path,required=True);p.add_argument('--wait-higher',type=int,default=0);a=p.parse_args()
 if a.password_file.stat().st_mode&0o077:raise SystemExit('Password file must be private (0600).')
 if not 0<=a.wait_higher<=180:raise SystemExit('--wait-higher must be 0..180 seconds')
 with httpx.Client(base_url=a.base_url,trust_env=False,timeout=30,headers={'X-Mandol-Client':'studio'}) as c:
  def call(method,path,**kw):
   r=c.request(method,path,**kw);r.raise_for_status();return r.json()
  identity=call('POST','/api/auth/login',json={'username':a.username,'password':a.password_file.read_text().strip()});c.headers['X-CSRF-Token']=identity['csrf_token']
  w=call('POST','/api/workspaces',json={'name':'Synthetic lifecycle example'})['id'];s=call('POST','/api/sessions',json={'workspace_id':w,'title':'A fictional notebook','unified':True})['id']
  scope={'workspace_id':w}
  def wait(receipt):
   deadline=time.monotonic()+30
   while time.monotonic()<deadline:
    if call('GET','/api/ingestions/'+receipt['ingestion_id'],params=scope)['status']=='saved':return
    time.sleep(.25)
   raise SystemExit('Ingestion not finished. Inspect Background Jobs; no input was declared saved.')
  payload={'workspace_id':w,'session_id':s,'request_id':str(uuid.uuid4()),'messages':[{'role':'user','content':'In this synthetic example, I prefer notebooks with blue covers.'}]}
  receipt=call('POST','/api/memories',json=payload);wait(receipt);uid=receipt['memory_ids'][0]
  original=call('GET','/api/memories/'+uid,params=scope)
  # A repeat request returns the same receipt and logical source.
  assert call('POST','/api/memories',json=payload)['memory_ids']==receipt['memory_ids']
  end=time.monotonic()+a.wait_higher;relations={}
  while True:
   relations=call('GET','/api/memories/'+uid+'/relations',params=scope)
   if a.wait_higher==0 or time.monotonic()>=end:break
   time.sleep(1)
  updated=call('PUT','/api/memories/'+uid,params=scope,json={'request_id':str(uuid.uuid4()),'expected_version':original['version'],'content':'Correction to the synthetic example: I now prefer green notebook covers.'});wait(updated)
  current=call('GET','/api/memories/'+uid,params=scope);assert current['version']==original['version']+1
  report={'workspace_id':w,'session_id':s,'source_uid':uid,'versions':[original['version'],current['version']],
          'provenance_before':relations,'provenance_after':call('GET','/api/memories/'+uid+'/relations',params={**scope,'state_view':'all'}),
          'history':call('GET','/api/memories/'+uid+'/history',params=scope),
          'jobs':[{k:j.get(k) for k in ['kind','status','error_code']} for j in call('GET','/api/jobs',params=scope)['items']]}
  print(json.dumps(report,ensure_ascii=False,indent=2));call('POST','/api/auth/logout',json={})
if __name__=='__main__':main()
