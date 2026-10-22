import {useEffect,useRef,useState} from 'react';
import {AlertTriangle,X} from 'lucide-react';
import {api,body,requestId} from '../api';
import type {Workspace} from '../types';
import {ErrorBox,Spinner} from '../ui';

const CONFIRMATION='清除所有记忆';
type ClearResult={workspace_id:string;records:number;basic:number;higher:number;status:string};

export default function ClearMemories({workspace,onClose,onCleared}:{workspace:Workspace;onClose:()=>void;onCleared:(count:number)=>void}){
 const dialog=useRef<HTMLDialogElement>(null),input=useRef<HTMLInputElement>(null),mounted=useRef(true),pending=useRef(false),id=useRef(requestId());
 const [confirmation,setConfirmation]=useState(''),[busy,setBusy]=useState(false),[error,setError]=useState('');
 useEffect(()=>{mounted.current=true;dialog.current?.showModal();input.current?.focus();return()=>{mounted.current=false}},[]);
 function close(){if(!pending.current)onClose()}
 async function clear(){
  if(confirmation!==CONFIRMATION||pending.current)return;
  pending.current=true;setBusy(true);setError('');
  try{
   const result=await api<ClearResult>('/workspaces/'+encodeURIComponent(workspace.id)+'/memories/clear',body({confirmation,request_id:id.current}));
   window.dispatchEvent(new CustomEvent('mandol-memories-cleared',{detail:{workspaceId:workspace.id}}));
   window.dispatchEvent(new Event('mandol-memory-changed'));
   if(mounted.current)onCleared(result.records);
  }catch(e){if(mounted.current)setError((e as Error).message)}
  finally{pending.current=false;if(mounted.current)setBusy(false)}
 }
 return <dialog ref={dialog} className="clear-memory-dialog" aria-labelledby="clear-memory-title" aria-describedby="clear-memory-description" aria-busy={busy} onCancel={e=>{e.preventDefault();close()}} onClick={e=>{if(e.target===e.currentTarget)close()}}>
  <form onSubmit={e=>{e.preventDefault();void clear()}} onClick={e=>e.stopPropagation()}>
   <header><h2 id="clear-memory-title"><AlertTriangle size={20}/>清除所有记忆</h2><button type="button" className="icon-button" aria-label="关闭清除确认" disabled={busy} onClick={close}><X size={19}/></button></header>
   <div id="clear-memory-description"><p>将清除工作区 <strong>{workspace.name}</strong> 中所有对话的基础记忆、高阶记忆和仓库知识，而不只是当前筛选结果。</p><p>会话历史、已上传附件和仓库配置保留，其他工作区及账号不受影响。删除审计仍可追溯；清除无法在界面撤销，后续对话可以继续生成新记忆。</p></div>
   <label className="clear-memory-label" htmlFor="clear-memory-confirmation">请输入 <strong>{CONFIRMATION}</strong> 确认</label>
   <input ref={input} id="clear-memory-confirmation" autoFocus autoComplete="off" spellCheck={false} value={confirmation} disabled={busy} onChange={e=>setConfirmation(e.target.value)} placeholder={CONFIRMATION}/>
   {error&&<ErrorBox error={error}/>}
   <footer><button className="secondary-button" type="button" disabled={busy} onClick={close}>取消</button><button className="clear-memory-confirm" type="submit" disabled={busy||confirmation!==CONFIRMATION}>{busy?<><Spinner/>正在清除…</>:'确认清除'}</button></footer>
  </form>
 </dialog>
}
