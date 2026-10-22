import {useEffect,useState} from 'react';

let csrfToken='';
export function resetAuth(){csrfToken=''}
export function authHeaders(){return {'X-Mandol-Client':'studio','X-CSRF-Token':csrfToken}}

export async function api<T>(path:string, options:RequestInit={}):Promise<T>{
  const response=await fetch('/api'+path,{credentials:'same-origin',...options,headers:{...authHeaders(),...(!(options.body instanceof FormData)?{'Content-Type':'application/json'}:{}),...options.headers}});
  const data=await response.json();
  if(response.status===401&&!path.startsWith('/auth/'))window.dispatchEvent(new Event('mandol-auth-expired'));
  if(!response.ok)throw new Error(data.error?.message||data.detail?.[0]?.msg||`请求失败 (${response.status})`);
  if(data.csrf_token)csrfToken=data.csrf_token;
  return data as T;
}
export const body=(data:unknown)=>({method:'POST',body:JSON.stringify(data)});
export function uploadFile<T>(path:string,data:FormData,signal:AbortSignal,onProgress:(percent:number)=>void):Promise<T>{
  return new Promise((resolve,reject)=>{
    const xhr=new XMLHttpRequest();xhr.open('POST','/api'+path);xhr.withCredentials=true;xhr.timeout=60000;
    Object.entries(authHeaders()).forEach(([key,value])=>xhr.setRequestHeader(key,value));
    const abort=()=>xhr.abort();signal.addEventListener('abort',abort,{once:true});
    const clean=()=>signal.removeEventListener('abort',abort);
    xhr.upload.onprogress=e=>{if(e.lengthComputable)onProgress(Math.round(e.loaded/e.total*100))};
    xhr.onerror=()=>{clean();reject(new Error('附件上传连接失败，请重试'))};
    xhr.ontimeout=()=>{clean();reject(new Error('附件上传超时，请核实后重试'))};
    xhr.onabort=()=>{clean();reject(new DOMException('上传已取消','AbortError'))};
    xhr.onload=()=>{clean();try{const value=JSON.parse(xhr.responseText);if(xhr.status===401)window.dispatchEvent(new Event('mandol-auth-expired'));if(xhr.status<200||xhr.status>=300)reject(new Error(value.error?.message||'附件校验或保存失败'));else resolve(value)}catch{reject(new Error('附件服务返回无效响应'))}};
    if(signal.aborted){clean();reject(new DOMException('上传已取消','AbortError'));return}xhr.send(data);
  });
}
export const requestId=()=>crypto.randomUUID();
export const qs=(data:Record<string,string|number|undefined>)=>new URLSearchParams(Object.entries(data).filter(([,v])=>v!==undefined&&v!=='').map(([k,v])=>[k,String(v)])).toString();
// Callers may retain results across search/pagination within an explicit scope.
// The default is the complete URL: IDs, permissions and representation changes
// still discard previous results synchronously, before an effect can run.
export function useResource<T>(path:string|null, tick=0, retainKey:string|null=path){
  const key=path===null?null:retainKey??path;
  const [state,setState]=useState<{key:string|null;path:string|null;data:T|null;error:string;loading:boolean}>({key:null,path:null,data:null,error:'',loading:false});
  const [refresh,setRefresh]=useState(0);
  useEffect(()=>{
    if(!path){setState({key:null,path:null,data:null,error:'',loading:false});return;}
    const ctrl=new AbortController();
    setState(previous=>({key,path,data:previous.key===key?previous.data:null,error:'',loading:true}));
    api<T>(path,{signal:ctrl.signal}).then(value=>{
      if(!ctrl.signal.aborted)setState({key,path,data:value,error:'',loading:false});
    }).catch(err=>{
      if(!ctrl.signal.aborted)setState(previous=>({...previous,error:err.message,loading:false}));
    });
    return ()=>ctrl.abort();
  },[path,key,tick,refresh]);
  const sameScope=path!==null&&state.key===key;
  return {data:sameScope?state.data:null,error:sameScope&&state.path===path?state.error:'',loading:path!==null&&(!sameScope||state.path!==path||state.loading),reload:()=>setRefresh(v=>v+1)};
}
export async function streamChat(data:unknown,signal:AbortSignal,onEvent:(name:string,data:any)=>void){
  const response=await fetch('/api/chat',{...body(data),credentials:'same-origin',headers:{'Content-Type':'application/json',...authHeaders()},signal});
  if(!response.ok){const error=await response.json();throw new Error(error.error?.message||'回答请求失败');}
  if(!response.body)throw new Error('浏览器不支持流式响应');
  const reader=response.body.getReader(),decoder=new TextDecoder();let buffer='';
  while(true){const {done,value}=await reader.read();if(done)break;buffer+=decoder.decode(value,{stream:true});
    let index;while((index=buffer.indexOf('\n\n'))>=0){const block=buffer.slice(0,index);buffer=buffer.slice(index+2);const name=block.split('\n').find(x=>x.startsWith('event: '))?.slice(7)||'message';const raw=block.split('\n').filter(x=>x.startsWith('data: ')).map(x=>x.slice(6)).join('\n');if(raw)onEvent(name,JSON.parse(raw));}
  }
}
