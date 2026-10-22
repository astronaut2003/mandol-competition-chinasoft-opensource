#!/usr/bin/env python3
"""Real GPU HTTP/protocol/stability check with bounded synthetic inputs.

No mock transport, model substitution, database writes or production data.
Run with the application .venv after starting your three independent services.
"""
from pathlib import Path
import argparse,json,math,statistics,sys,time
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mandol_competition.chinasoft.config import Settings
from mandol_competition.chinasoft.native_retrieval import NativeRetrievalModels
from mandol_competition.chinasoft.retrieval_contract import MODELS,recipe_hash

def percentile(values,p):
 ordered=sorted(values);i=(len(ordered)-1)*p;lo=int(i);hi=math.ceil(i)
 return round(ordered[lo]+(ordered[hi]-ordered[lo])*(i-lo),3)

def verify(samples,channels):
 client=NativeRetrievalModels(Settings(),None);out={'kind':'real-gpu-http','samples_per_channel':samples,'channels':channels,'checks':[],'health':{},'latency_ms':{}}
 try:
  out['health']={c:client.probe(c,force=True) for c in channels}
  if not all(h.get('identity_ready') for h in out['health'].values()):raise RuntimeError('Selected services must pass actual identity and inference checks: '+json.dumps(out['health']))
  docs=['The camera is inside the blue cabinet.','Bananas grow in warm weather.']
  if 'dense' in channels:
   dense,_=client.dense(docs,'document');again,_=client.dense(docs,'document')
   assert len(dense)==2 and all(len(v)==1024 and abs(math.hypot(*v)-1)<.02 for v in dense)
   stable=max(abs(a-b) for x,y in zip(dense,again) for a,b in zip(x,y));assert stable<.005
   raw=client.request('dense','POST','/v1/embeddings',{'model':MODELS['dense']['alias'],'input':docs,'encoding_format':'float','truncate_prompt_tokens':4096,'truncation_side':'right'})
   batch=client.ordered('dense',raw,2,'embedding');batch_error=max(abs(a-b) for x,y in zip(dense,batch) for a,b in zip(x,y));assert batch_error<.02
   out['checks'].append({'name':'dense_real_1024_normalized_batch_stable','status':'PASS','norms':[math.hypot(*v) for v in dense],'repeat_max_abs':stable,'batch_max_abs':batch_error})
  if 'sparse' in channels:
   sparse=client.sparse_batch(docs,'document');again=client.sparse_batch(docs,'query')
   assert all(v and all(0<=i<30522 and math.isfinite(x) and x>0 for i,x in v.items()) for v in sparse)
   assert all(set(a)==set(b) and max(abs(a[i]-b[i]) for i in a)<.02 for a,b in zip(sparse,again))
   raw=client.request('sparse','POST','/pooling',{'model':MODELS['sparse']['alias'],'input':docs,'task':'embed','encoding_format':'float','add_special_tokens':True,'truncate_prompt_tokens':512,'truncation_side':'right','use_activation':False})
   sbatch=client.ordered('sparse',raw,2,'data');assert all(len(v)==30522 for v in sbatch)
   error=max(abs(sparse[j].get(i,0)-value) for j,vector in enumerate(sbatch) for i,value in enumerate(vector));assert error<.08
   out['checks'].append({'name':'splade_real_vocabulary_batch_query_document','status':'PASS','vocabulary':30522,'nonzero':[len(v) for v in sparse],'batch_max_abs':error})
  if 'reranker' in channels:
   ranked=client.rerank('Where is the camera?',docs);assert dict(ranked)[0]>dict(ranked)[1];assert client.rerank('test',[])==[]
   out['checks'].append({'name':'bge_real_pairs_sigmoid_orientation','status':'PASS','scores':ranked})
  for channel in channels:
   meta=client.request(channel,'GET','/mandol/metadata');assert meta['recipe_hash']==recipe_hash(channel)
   out.setdefault('metadata',{})[channel]=meta
   def call(_):
    start=time.perf_counter()
    if channel=='dense':client.dense([docs[0]],'query')
    elif channel=='sparse':client.sparse_batch([docs[0]],'query')
    else:client.rerank('Where is the camera?',docs)
    return (time.perf_counter()-start)*1000
   sequential=[call(i) for i in range(samples)]
   with ThreadPoolExecutor(max_workers=2) as pool:concurrent=list(pool.map(call,range(samples)))
   out['latency_ms'][channel]={'sequential':sequential,'concurrency_2':concurrent,'p50':percentile(sequential,.5),'p95':percentile(sequential,.95),'concurrent_p50':percentile(concurrent,.5),'concurrent_p95':percentile(concurrent,.95)}
  out['health']={c:client.probe(c,force=True) for c in channels}
  assert all(h.get('ready') for h in out['health'].values())
  out['status']='PASS';return out
 finally:client.close()

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--channel',choices=list(MODELS));p.add_argument('--samples',type=int,default=12);p.add_argument('--output',type=Path);a=p.parse_args()
 if not 2<=a.samples<=50:p.error('--samples must be 2..50')
 try:report=verify(a.samples,[a.channel] if a.channel else list(MODELS))
 except Exception as error:report={'kind':'real-gpu-http','status':'FAIL','error_type':type(error).__name__,'message':str(error)[:1800]}
 text=json.dumps(report,ensure_ascii=False,indent=2)+'\n'
 if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(text)
 else:print(text)
 if report['status']!='PASS':raise SystemExit(1)
if __name__=='__main__':main()
