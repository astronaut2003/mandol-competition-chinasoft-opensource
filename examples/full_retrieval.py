#!/usr/bin/env python3
"""Real-model memory E2E in a NEW synthetic workspace; no mocks or substitutions.

Requires the three qualified GPU HTTP services plus a configured generation API.
Run with the application .venv. Creates a small workspace and keeps it for review.
"""
from pathlib import Path
import argparse,json,time,uuid
import httpx

def run(base,username,password,report):
 with httpx.Client(base_url=base,timeout=180,trust_env=False,headers={'X-Mandol-Client':'studio'}) as client:
  def req(method,path,**kw):
   response=client.request(method,path,**kw);response.raise_for_status();return response.json()
  def check(name,condition,**evidence):
   report['checks'].append({'name':name,'status':'PASS' if condition else 'FAIL',**evidence})
   if not condition:raise AssertionError(name)
  identity=req('POST','/api/auth/login',json={'username':username,'password':password})
  client.headers['X-CSRF-Token']=identity['csrf_token']
  initial=identity['workspaces'][0]['id']
  health=req('GET','/api/status',params={'workspace_id':initial})
  models=health['retrieval'].get('models',{})
  check('all_three_models_ready',all(models.get(k,{}).get('ready') for k in ('dense','sparse','reranker')),models=models)
  workspace=req('POST','/api/workspaces',json={'name':'GPU E2E '+uuid.uuid4().hex[:8]})['id'];report['workspace_id']=workspace
  def session(title):return req('POST','/api/sessions',json={'workspace_id':workspace,'title':title,'unified':True})['id']
  first=session('Synthetic notebook preferences')
  payload={'workspace_id':workspace,'session_id':first,'request_id':str(uuid.uuid4()),'messages':[
   {'role':'user','content':'I use a blue notebook for work. I prefer it because the cover is easy to find on my desk.'},
   {'role':'user','content':'For my weekend sketches and new ideas I use a green notebook. This is a separate preference from my work notebook.'},
   {'role':'user','content':'My small code reading tool has a retry budget of three attempts. I record debugging decisions in my work notebook.'}]}
  receipt=req('POST','/api/memories',json=payload);report['receipt']=receipt
  check('idempotent_durable_receipt',receipt==req('POST','/api/memories',json=payload))
  def jobs_done(timeout=360):
   deadline=time.monotonic()+timeout
   while time.monotonic()<deadline:
    jobs=req('GET','/api/jobs',params={'workspace_id':workspace,'limit':50})['items']
    if jobs and all(j['status'] in ('completed','failed','cancelled') for j in jobs):
     return jobs
    time.sleep(1)
   raise TimeoutError('Background tasks did not finish within the bounded verification timeout')
  jobs=jobs_done();report['initial_jobs']=jobs
  check('real_index_and_derivation_completed',any(j['kind']=='index' and j['status']=='completed' for j in jobs) and any(j['kind']=='derive' and j['status']=='completed' for j in jobs))
  memories=req('GET','/api/memories',params={'workspace_id':workspace,'retrieval_scope':'conversations','limit':50})['items']
  higher=[m for m in memories if m['source_kind']=='derived' and m['eligible']]
  check('actual_higher_memories_with_sources',bool(higher) and all(m.get('source_versions') for m in higher),higher=[{'id':m['id'],'type':m['memory_type'],'version':m['version'],'source_versions':m['source_versions']} for m in higher])
  second=session('Cross-conversation recall');report['query_session_id']=second
  query='From our earlier conversations, what color is the notebook I use for work?'
  result=req('GET','/api/memories/search',params={'workspace_id':workspace,'session_id':second,'retrieval_scope':'conversations','q':query,'limit':8})
  trace=result.get('route',{}).get('trace',{});report['retrieval']=result
  check('dense_splade_bm25_rrf_bge',all(trace.get('candidate_counts',{}).get(k,0)>0 for k in ('cosine','splade','bm25')) and trace.get('three_channels_consumed') and trace.get('reranker_used'),trace=trace)
  source_ids=set(receipt['memory_ids'])
  cited_sources={s['id'] for m in result['items'] for s in m.get('evidence_sources',[])}
  check('higher_to_valid_basic_evidence',bool(cited_sources&source_ids))
  start=time.perf_counter();first_token=None;answer='';events=[];sources=[];route={};errors=[]
  chat_payload={'workspace_id':workspace,'session_id':second,'request_id':str(uuid.uuid4()),'content':query,'retrieval_scope':'conversations'}
  def chat(payload):
   nonlocal first_token,answer,sources,route
   seen=[]
   with client.stream('POST','/api/chat',json=payload) as response:
    response.raise_for_status();event=''
    for line in response.iter_lines():
     if line.startswith('event:'):event=line[6:].strip()
     if line.startswith('data:'):
      data=json.loads(line[5:]);seen.append((event,data))
      if event=='token':
       if first_token is None:first_token=(time.perf_counter()-start)*1000
       answer+=data['text']
      if event=='sources':sources=data
      if event=='route':route=data
      if event=='error':errors.append(data)
   return seen
  events=chat(chat_payload)
  report['generation']={'answer':answer,'sources':sources,'route':route,'errors':errors,'first_token_ms':first_token,'total_ms':(time.perf_counter()-start)*1000}
  check('real_agent_answer_and_citations',not errors and any(k=='done' for k,_ in events) and ('blue' in answer.lower() or '蓝' in answer) and bool(sources))
  replay=chat(chat_payload)
  check('completed_request_replay_cached',any(k=='done' and d.get('cached') for k,d in replay))
  basic=req('GET','/api/memories/'+receipt['memory_ids'][0],params={'workspace_id':workspace})
  revision=req('PUT','/api/memories/'+basic['id'],params={'workspace_id':workspace},json={'request_id':str(uuid.uuid4()),'expected_version':basic['version'],'content':'I now use a red notebook for work. The old blue notebook has been retired from work use.'})
  jobs=jobs_done();new=req('GET','/api/memories/'+basic['id'],params={'workspace_id':workspace})
  check('basic_source_version_increments',new['version']==basic['version']+1)
  affected=[m for m in higher if basic['id'] in m['source_versions']]
  audited=[req('GET','/api/memories/'+m['id'],params={'workspace_id':workspace,'audit':True}) for m in affected]
  check('old_higher_sources_invalidated',bool(affected) and all(not m['eligible'] for m in audited),affected_ids=[m['id'] for m in audited])
  relations=req('GET','/api/memories/'+basic['id']+'/relations',params={'workspace_id':workspace,'state_view':'all'})
  report['revision']={'receipt':revision,'version':new['version'],'affected':audited,'relations':relations,'jobs':jobs}
  check('basic_to_higher_relations_available',bool(relations.get('items')))
  history=req('GET','/api/memories/'+basic['id']+'/history',params={'workspace_id':workspace})
  report['revision']['history']=history
  check('saved_source_versions_remain_traceable',{basic['version'],new['version']}.issubset({v['version'] for v in history.get('items',[])}))
  final=req('GET','/api/memories/search',params={'workspace_id':workspace,'session_id':second,'retrieval_scope':'conversations','q':query,'limit':8})
  invalid_ids={m['id'] for m in audited}
  check('invalid_higher_excluded_from_search',not any(m['id'] in invalid_ids for m in final['items']))
  report['post_revision_retrieval']=final
  report['status']='PASS';return report

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--url',default='http://127.0.0.1:20600');parser.add_argument('--username',required=True);parser.add_argument('--password-file',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
 if a.password_file.stat().st_mode&0o077:parser.error('Password file must have mode 0600')
 report={'kind':'real-memory-gpu-agent-e2e','checks':[],'status':'FAIL'}
 try:run(a.url,a.username,a.password_file.read_text().strip(),report)
 except Exception as error:report.update(status='FAIL',error_type=type(error).__name__,message=str(error)[:1800])
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 print(report['status'])
 if report['status']!='PASS':raise SystemExit(1)
if __name__=='__main__':main()
