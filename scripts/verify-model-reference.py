#!/usr/bin/env python3
"""Compare a real vLLM HTTP output with same-checkpoint Transformers CUDA math.

Run in the channel's independent GPU environment, never the Web application.
The reference loads only the explicitly configured, hash-verified public cache.
"""
from pathlib import Path
import argparse,json,os,sys,urllib.request
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'packages/substrate/src')]
from mandol_competition.chinasoft.retrieval_contract import MODELS,cache_snapshot,verify_artifact,recipe_hash
from mandol_competition.chinasoft.config import Settings

def verify(channel):
 import torch
 import torch.nn.functional as F
 from transformers import AutoTokenizer,AutoModel,AutoModelForMaskedLM,AutoModelForSequenceClassification
 if not torch.cuda.is_available():raise RuntimeError('CUDA is required; there is no CPU substitute in this verification')
 cache=Path(os.environ.get('CHINASOFT_MODEL_CACHE',ROOT/'data/models/hub')).resolve()
 artifact=verify_artifact(cache,channel);snapshot=cache_snapshot(cache,channel);model=MODELS[channel];settings=Settings()
 base={'dense':settings.embedding_url,'sparse':settings.sparse_url,'reranker':settings.reranker_url}[channel].rstrip('/').removesuffix('/v1')
 key_name={'dense':settings.embedding_api_key_env,'sparse':settings.sparse_api_key_env,'reranker':settings.reranker_api_key_env}[channel]
 def request(path,body=None):
  headers={'Content-Type':'application/json'}
  if key_name:headers['Authorization']='Bearer '+os.environ[key_name]
  req=urllib.request.Request(base+path,None if body is None else json.dumps(body).encode(),headers=headers)
  with urllib.request.urlopen(req,timeout=90) as r:return json.load(r)
 meta=request('/mandol/metadata');assert meta['recipe_hash']==recipe_hash(channel)
 tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False)
 docs=['The camera is inside the blue cabinet.','Bananas grow in warm weather.'];query='Where is the camera?'
 if channel=='dense':
  net=AutoModel.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16).to('cuda').eval()
  inputs=tokenizer(docs,padding=True,truncation=True,max_length=4096,return_tensors='pt').to('cuda')
  with torch.inference_mode():
   hidden=net(**inputs).last_hidden_state
   indexes=inputs['attention_mask'].sum(dim=1)-1
   reference=F.normalize(hidden[torch.arange(len(docs),device='cuda'),indexes].float(),p=2,dim=1).cpu()
  raw=request('/v1/embeddings',{'model':model['alias'],'input':docs,'encoding_format':'float','truncate_prompt_tokens':4096,'truncation_side':'right'})
  actual=torch.tensor([x['embedding'] for x in sorted(raw['data'],key=lambda x:x['index'])])
 elif channel=='sparse':
  net=AutoModelForMaskedLM.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16).to('cuda').eval()
  inputs=tokenizer(docs,padding=True,truncation=True,max_length=512,return_tensors='pt').to('cuda')
  with torch.inference_mode():
   logits=net(**inputs).logits.float()
   reference=(torch.log1p(torch.relu(logits))*inputs['attention_mask'].unsqueeze(-1)).max(dim=1).values.cpu()
  raw=request('/pooling',{'model':model['alias'],'input':docs,'task':'embed','encoding_format':'float','add_special_tokens':True,'truncate_prompt_tokens':512,'truncation_side':'right','use_activation':False})
  actual=torch.tensor([x['data'] for x in sorted(raw['data'],key=lambda x:x['index'])])
  assert actual.shape==(2,30522)
 else:
  net=AutoModelForSequenceClassification.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16).to('cuda').eval()
  inputs=tokenizer([query]*2,docs,padding=True,truncation=True,max_length=4096,return_tensors='pt').to('cuda')
  with torch.inference_mode():reference=net(**inputs).logits.float().reshape(-1).sigmoid().cpu()
  raw=request('/score',{'model':model['alias'],'text_1':query,'text_2':docs,'truncate_prompt_tokens':4096,'truncation_side':'right','use_activation':True})
  actual=torch.tensor([x['score'] for x in sorted(raw['data'],key=lambda x:x['index'])]);assert actual[0]>actual[1]
 error=float((actual-reference).abs().max());cosine=float(F.cosine_similarity(actual.reshape(1,-1),reference.reshape(1,-1)).item())
 limit={'dense':.03,'sparse':.15,'reranker':.03}[channel]
 report={'channel':channel,'kind':'real-transformers-cuda-reference','status':'PASS' if error<limit and cosine>.995 else 'FAIL','max_absolute_error':error,'cosine':cosine,'absolute_error_limit':limit,'cosine_minimum':.995,'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'cuda':torch.version.cuda,'shape':list(actual.shape),'weights_sha256':artifact['weights_sha256'],'recipe_hash':recipe_hash(channel),'revision':model['revision']}
 if channel=='sparse':report['reference_math']='max over non-padding tokens of log1p(ReLU(masked-LM logits)); special tokens included'
 return report

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--channel',required=True,choices=MODELS);p.add_argument('--output',type=Path);a=p.parse_args()
 try:report=verify(a.channel)
 except Exception as error:report={'channel':a.channel,'status':'FAIL','error_type':type(error).__name__,'message':str(error)[:1800]}
 text=json.dumps(report,ensure_ascii=False,indent=2)+'\n'
 if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(text)
 else:print(text)
 if report['status']!='PASS':raise SystemExit(1)
if __name__=='__main__':main()
