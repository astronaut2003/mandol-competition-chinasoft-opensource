import {useState,useEffect,useRef,useSyncExternalStore,memo,lazy,Suspense} from 'react';
import {ArrowUp,Plus,Square,Save,X,ArrowRight,FileText,Database,Globe} from 'lucide-react';
import {api,body,qs,requestId,useResource,streamChat,uploadFile} from '../api';
import type {PageProps,ChatRecord,Memory,SourceLocator,WebSource} from '../types';
import RepositorySelect from '../RepositorySelect';
import {Badge,Spinner,ErrorBox} from '../ui';
const MarkdownAnswer=lazy(()=>import('./MarkdownAnswer'));
const RetrievalExplain=lazy(()=>import('./RetrievalExplain'));
function textStore(){let text='';const listeners=new Set<()=>void>();return {get:()=>text,set:(v:string)=>{text=v;listeners.forEach(f=>f())},subscribe:(f:()=>void)=>{listeners.add(f);return()=>{listeners.delete(f)}}}}
const StreamText=memo(function StreamText({store}:{store:ReturnType<typeof textStore>}){const text=useSyncExternalStore(store.subscribe,store.get);return <p className="answer-text streaming-answer">{text}</p>});
const Photos=memo(function Photos({workspaceId,ids}:{workspaceId:string;ids:string[]}){return <div className="message-photos">{ids.map(id=><a key={id} href={'/api/media/'+id+'?'+qs({workspace_id:workspaceId})} target="_blank" rel="noreferrer"><img src={'/api/media/'+id+'?'+qs({workspace_id:workspaceId,thumb:1})} loading="lazy" alt="原图证据"/></a>)}</div>});
type Attachment={id:string;name:string;url?:string;image:boolean;version?:number;parser?:string;bytes?:number};
type FilePreview={id:string;name:string;text:string;parser:string;version:number;active:number};
function Evidence({sources,files,onDetail,onFile}:{sources:ChatRecord['citations'];files:SourceLocator[];onDetail:(id:string)=>void;onFile:(id:string,startLine?:number)=>void}){
 if(!sources.length&&!files.length)return null;
 return <details className="reference-memories"><summary>查看依据 · {sources.length>0&&`${sources.length} 条参考记忆`}{!!sources.length&&!!files.length&&' / '}{files.length>0&&`${files.length} 个文件片段`}</summary><div className="citation-list">{sources.map((m,i)=><button key={m.id} onClick={()=>onDetail(m.id)}><span>{m.citation_index||i+1}</span>{m.source_locator?`${m.source_locator.path}:${m.source_locator.start_line}-${m.source_locator.end_line}`:m.text.slice(0,38)}</button>)}{files.map(f=><button key={f.attachment_id} onClick={()=>f.attachment_id&&onFile(f.attachment_id,f.start_line)}><FileText size={13}/>{f.path}:{f.start_line}-{f.end_line} · v{f.version||1}</button>)}</div></details>
}
function WebEvidence({route}:{route?:Record<string,any>}){
 const web=route?.web,sources=(route?.web_sources||[]) as WebSource[];
 if(!web?.requested)return null;
 return <details className="web-evidence"><summary><Globe size={13}/>联网{web.status==='processing'?'处理中':web.status==='interrupted'?'已中断 · 远程结果未确认':web.status==='failed'?'未成功':web.status==='skipped'?'未执行 · 本轮无需搜索':web.status==='empty'?'无结果':sources.length?` · ${sources.length} 个网页来源`:'已执行 · 未引用网页'}</summary>{(web.status==='failed'||web.status==='interrupted')&&<p role="status">{web.message}；本轮没有网页依据。</p>}{sources.map(s=><article key={s.id}><a href={s.url} target="_blank" rel="noreferrer noopener"><strong>[{s.id}] {s.title}</strong></a><small>{s.site_name} · {s.published_at?`发布 ${new Date(s.published_at).toLocaleDateString()} · `:''}检索 {new Date(s.retrieved_at).toLocaleString()}</small><p>{s.excerpt.slice(0,500)}</p></article>)}{web.warnings?.length>0&&<p>部分抓取未完成：{web.warnings.join('、')}</p>}</details>
}
export default function Chat(props:PageProps & {visual?:boolean}){
 const {workspace,session,onDetail,onActivity}=props;
 const [webSearch,setWebSearch]=useState(false),[reasoning,setReasoning]=useState('fast');
 const capabilities=useResource<{generation:{modes:string[];reason:string};web:{enabled:boolean;configured:boolean;privacy:string}}>('/chat/capabilities');
 const draftKey=`mandol:composer:${workspace.id}:${session?.id||'draft'}`;
 const [text,setText]=useState(()=>{try{return sessionStorage.getItem(draftKey)||''}catch{return ''}}),[attachments,setAttachments]=useState<Attachment[]>([]),[pending,setPending]=useState(false),[uploading,setUploading]=useState(false),[uploadProgress,setUploadProgress]=useState(''),[error,setError]=useState(''),[status,setStatus]=useState(''),[sources,setSources]=useState<ChatRecord['citations']>([]),[fileSources,setFileSources]=useState<SourceLocator[]>([]),[receipt,setReceipt]=useState<any>(null),[liveRoute,setLiveRoute]=useState<Record<string,any>|undefined>(),[liveQuestion,setLiveQuestion]=useState(''),[offset,setOffset]=useState(0),[scope,setScope]=useState('conversation'),[knowledgeOpen,setKnowledgeOpen]=useState(false),[preview,setPreview]=useState<FilePreview|null>(null),[previewError,setPreviewError]=useState(''),[previewOffset,setPreviewOffset]=useState(0);
 const history=useResource<{items:ChatRecord[]}> (session?'/chat/history?'+qs({workspace_id:workspace.id,session_id:session.id,offset,limit:40}):null);
 const file=useRef<HTMLInputElement>(null),input=useRef<HTMLTextAreaElement>(null),scroll=useRef<HTMLDivElement>(null),abort=useRef<AbortController|null>(null),uploadAbort=useRef<AbortController|null>(null),previewAbort=useRef<AbortController|null>(null),generation=useRef(0),timer=useRef<ReturnType<typeof setTimeout>|null>(null),stream=useRef(textStore()),nearBottom=useRef(true),alive=useRef(true),attachmentsRef=useRef(attachments),pendingRef=useRef(false),lastSave=useRef<any>(null),lastChat=useRef<any>(null);
 attachmentsRef.current=attachments;
 const uploadingRef=useRef(false),pasteEvents=useRef(new WeakSet<Event>()),clipboardSequence=useRef(0),uploadUrl=useRef('');
 const [failedUpload,setFailedUpload]=useState<File|null>(null),[uploadPreview,setUploadPreview]=useState<{name:string;url:string}|null>(null);
 const uploadQueue=useRef<File[]>([]),[queuedUploads,setQueuedUploads]=useState(0);
 useEffect(()=>{alive.current=true;return()=>{alive.current=false;generation.current++;abort.current?.abort();uploadAbort.current?.abort();previewAbort.current?.abort();if(timer.current)clearTimeout(timer.current);if(uploadUrl.current)URL.revokeObjectURL(uploadUrl.current);attachmentsRef.current.forEach(a=>a.url&&URL.revokeObjectURL(a.url))}},[]);
 useEffect(()=>{const area=input.current;if(area){area.style.height='0px';area.style.height=Math.min(240,Math.max(52,area.scrollHeight))+'px';area.style.overflowY=area.scrollHeight>240?'auto':'hidden'}try{if(text)sessionStorage.setItem(draftKey,text);else sessionStorage.removeItem(draftKey)}catch{}},[text,draftKey]);
 useEffect(()=>{if(offset===0&&nearBottom.current&&!pending)requestAnimationFrame(()=>scroll.current?.scrollTo({top:scroll.current.scrollHeight,behavior:'instant'}))},[history.data,offset,pending]);
 function release(){attachmentsRef.current.forEach(a=>a.url&&URL.revokeObjectURL(a.url));attachmentsRef.current=[];setAttachments([]);setText('');uploadQueue.current=[];setQueuedUploads(0);setFailedUpload(null)}
 async function showFile(id:string,startLine=1){setPreviewOffset(Math.floor(Math.max(0,startLine-1)/160)*160);previewAbort.current?.abort();const ctrl=new AbortController();previewAbort.current=ctrl;setPreviewError('');try{const value=await api<FilePreview>('/attachments/'+id+'?'+qs({workspace_id:workspace.id}),{signal:ctrl.signal});if(alive.current&&!ctrl.signal.aborted)setPreview(value)}catch(e){if(!ctrl.signal.aborted)setPreviewError((e as Error).message)}}
 async function withdrawFile(){if(!preview||!window.confirm('停止使用这个附件版本？基础文件与历史记录会保留，其高阶结果将退出当前检索。'))return;try{await api('/attachments/'+preview.id+'?'+qs({workspace_id:workspace.id}),{method:'DELETE'});setPreview({...preview,active:0});attachmentsRef.current=attachmentsRef.current.filter(a=>a.id!==preview.id);setAttachments(attachmentsRef.current);history.reload()}catch(e){setPreviewError((e as Error).message)}}
 async function upload(files:FileList|File[]){
  if(!session||pendingRef.current)return;
  const incoming=Array.from(files);
  const replacements=incoming.filter(f=>!/^image\//.test(f.type)&&attachmentsRef.current.some(a=>!a.image&&a.name===f.name)).length;
  if(incoming.length>4||attachmentsRef.current.length+uploadQueue.current.length+incoming.length+(uploadingRef.current?1:0)-replacements>4){setError('最多 4 个附件（包含正在上传和排队中的文件）');return}
  if(incoming.some(f=>f.size>(/^image\//.test(f.type)||/\.(png|jpe?g|webp)$/i.test(f.name)?8*1024*1024:256*1024))){setError('图片最大 8 MiB，文本附件最大 256 KiB');return}
  if(uploadingRef.current){uploadQueue.current.push(...incoming);setQueuedUploads(uploadQueue.current.length);return}
  uploadQueue.current.unshift(...incoming);uploadingRef.current=true;setUploading(true);setError('');setFailedUpload(null);let currentFile:File|null=null;const current=generation.current,ctrl=new AbortController();uploadAbort.current=ctrl;
  try{while(uploadQueue.current.length){const f=uploadQueue.current.shift()!;setQueuedUploads(uploadQueue.current.length);
   currentFile=f;
   const image=/^image\/(png|jpeg|webp)$/.test(f.type)||/\.(png|jpe?g|webp)$/i.test(f.name);
   const replacement=!image&&attachmentsRef.current.some(a=>!a.image&&a.name===f.name);
   if(attachmentsRef.current.length>=4&&!replacement)throw new Error('每次最多 4 个附件');
   if(f.size>(image?8*1024*1024:256*1024))throw new Error(image?'图片最大 8 MiB':'文本附件最大 256 KiB');
   const data=new FormData();data.set('workspace_id',workspace.id);data.set('session_id',session.id);data.set('file',f);
   setUploadProgress(`${f.name} · 上传中`);
   if(image){uploadUrl.current=URL.createObjectURL(f);setUploadPreview({name:f.name,url:uploadUrl.current})}
   const a=await uploadFile<Attachment>(image?'/media':'/attachments',data,ctrl.signal,p=>{if(alive.current)setUploadProgress(`${f.name} · ${p===100?'校验并保存中':`上传 ${p}%`}`)});
   if(!alive.current||current!==generation.current)return;
   const next={...a,name:f.name,image,url:image?uploadUrl.current:undefined};uploadUrl.current='';setUploadPreview(null);
   const retained=attachmentsRef.current.filter(old=>old.id!==a.id&&(image||old.image||old.name!==f.name));
   attachmentsRef.current.filter(old=>!retained.includes(old)).forEach(old=>old.url&&URL.revokeObjectURL(old.url));
   attachmentsRef.current=[...retained,next];setAttachments(attachmentsRef.current);
  }}catch(e){if(alive.current&&(e as Error).name!=='AbortError'){setError((e as Error).message);setFailedUpload(currentFile)}else uploadQueue.current=[]}finally{uploadingRef.current=false;if(uploadUrl.current){URL.revokeObjectURL(uploadUrl.current);uploadUrl.current=''}if(alive.current){setQueuedUploads(uploadQueue.current.length);setUploading(false);setUploadPreview(null);setUploadProgress('')}uploadAbort.current=null;if(file.current)file.current.value=''}
 }
 function paste(e:React.ClipboardEvent<HTMLDivElement>){
  const images=Array.from(e.clipboardData.items).filter(item=>item.kind==='file'&&/^image\//.test(item.type)).map(item=>item.getAsFile()).filter((f):f is File=>!!f);
  if(!images.length)return; // Plain text/code keeps its native browser insertion.
  e.preventDefault();if(pasteEvents.current.has(e.nativeEvent))return;pasteEvents.current.add(e.nativeEvent);
  const plain=e.clipboardData.getData('text/plain'),target=input.current;
  if(plain&&target){const start=target.selectionStart,end=target.selectionEnd;setText(value=>(value.slice(0,start)+plain+value.slice(end)).slice(0,8000));requestAnimationFrame(()=>{target.focus();target.setSelectionRange(Math.min(start+plain.length,8000),Math.min(start+plain.length,8000))})}
  if(images.some(f=>!/^image\/(png|jpeg|webp)$/.test(f.type))){setError('剪贴板仅支持 PNG、JPEG、WebP 图片');return}
  if(pendingRef.current){setError('请等待当前回答完成后再粘贴图片');return}
  const stamp=Date.now();upload(images.map(f=>new File([f],`clipboard-${stamp}-${++clipboardSequence.current}.${f.type==='image/jpeg'?'jpg':f.type.split('/')[1]}`,{type:f.type})));
 }
 async function save(retry=false){
  if(!session||pendingRef.current||uploadingRef.current||uploadQueue.current.length)return;
  const data=retry?lastSave.current:{workspace_id:workspace.id,session_id:session.id,mode:session.mode,repository_id:session.repository_id||null,request_id:requestId(),attachment_ids:attachments.filter(x=>!x.image).map(x=>x.id),messages:[{role:'user',content:text||'用户上传附件',media_ids:attachments.filter(x=>x.image).map(x=>x.id)}]};
  if(!data)return;lastSave.current=data;lastChat.current=null;pendingRef.current=true;setPending(true);setError('');
  try{const r=await api('/memories',body(data));if(!alive.current)return;setReceipt(r);release();onActivity?.();history.reload();lastSave.current=null}catch(e){if(alive.current)setError((e as Error).message)}finally{pendingRef.current=false;if(alive.current)setPending(false)}
 }
 async function ask(record?:ChatRecord,retry=false){
  if(!session||pendingRef.current||uploadingRef.current||uploadQueue.current.length||(!record&&!retry&&!text.trim()&&!attachmentsRef.current.length))return;
  const value=record?.question||text||'请帮我看看本次附件。';
  const savedOptions=record?.route?.options||{};
  const data=retry?lastChat.current:{workspace_id:workspace.id,session_id:session.id,request_id:record?.request_id||requestId(),mode:'auto',repository_id:session.repository_id||null,retrieval_scope:savedOptions.retrieval_scope||scope,content:value,media_ids:record?.media||attachments.filter(a=>a.image).map(a=>a.id),attachment_ids:record?.attachment_ids||attachments.filter(a=>!a.image).map(a=>a.id),reasoning_mode:record?(savedOptions.reasoning_mode||'fast'):reasoning,web_search:record?!!savedOptions.web_search:webSearch,...(record&&savedOptions.web_query?{web_query:savedOptions.web_query}:{})};
  if(!data)return;lastChat.current=data;lastSave.current=null;const current=++generation.current;pendingRef.current=true;setOffset(0);setPending(true);setError('');setLiveQuestion(data.content);nearBottom.current=true;setSources([]);setFileSources([]);setLiveRoute(undefined);setStatus('正在保存消息');stream.current.set('');
  const ctrl=new AbortController();abort.current=ctrl;let accumulated='',done=false,accepted=false;
  const flush=()=>{timer.current=null;if(!alive.current||current!==generation.current)return;stream.current.set(accumulated);if(nearBottom.current)scroll.current?.scrollTo({top:scroll.current.scrollHeight,behavior:'instant'})};
  try{
   await streamChat(data,ctrl.signal,(name,d)=>{
    if(!alive.current||current!==generation.current)return;
    if(name==='accepted'){accepted=true;setReceipt(d.receipt);if(!record&&!retry){release()}onActivity?.()}
    if(name==='status')setStatus(d.message);
    if(name==='sources')setSources(d);
    if(name==='route'){setFileSources(d.attachment_sources||[]);setLiveRoute(d)}
    if(name==='token'){accumulated+=d.text;if(!timer.current)timer.current=setTimeout(flush,64);setStatus('正在回答')}
    if(name==='error')throw new Error(d.message);
    if(name==='done'){done=true;lastChat.current=null}
   });
   if(!done&&!ctrl.signal.aborted)throw new Error('连接中断，可使用同一请求标识重试');
  }catch(e){if(alive.current&&current===generation.current)setError(ctrl.signal.aborted?(accepted?'已中断，用户消息已保存，可重试':'已中断；可用同一请求标识核实和重试'):(e as Error).message)}
  finally{if(timer.current)clearTimeout(timer.current);flush();pendingRef.current=false;if(alive.current&&current===generation.current){setPending(false);setLiveQuestion('');history.reload();onActivity?.();abort.current=null;input.current?.focus()}}
 }
 const questions=['今天想聊些什么？','帮我解释一下这段代码','我们之前做过哪些项目决策？'];
 return <div className="chat-page">
  <div className="page-heading"><div><span className="eyebrow">MEMORY-AWARE AGENT</span><h1>跨越会话，记住重要的事</h1><p>自然交流，按需回忆。文字、图片和代码都可以在这里讨论。</p></div><Badge>统一对话</Badge></div>
  <div className="chat-toolbar">
   <label>检索范围 <select aria-label="检索范围" value={scope} disabled={pending} onChange={e=>setScope(e.target.value)}><option value="conversation">本对话</option><option value="conversations">所有对话（当前工作区）</option></select></label>
   <details className="knowledge-sources" onToggle={e=>setKnowledgeOpen(e.currentTarget.open)}><summary><Database size={14}/>知识源{session?.repository_id&&<span className="quiet-tag">已关联仓库</span>}</summary>{knowledgeOpen&&<div className="knowledge-options"><p>默认使用当前附件与授权范围内的对话。完整仓库是可选的高级知识源。</p><RepositorySelect {...props}/><a href="#memories">查看记忆与来源</a></div>}</details>
  </div>
  <div ref={scroll} className="chat-scroll" onScroll={e=>{const t=e.currentTarget;nearBottom.current=t.scrollHeight-t.scrollTop-t.clientHeight<100}}>
   {((history.data?.items.length||0)>=40||offset>0)&&<div className="history-pagination"><button onClick={()=>setOffset(x=>x+40)} disabled={pending||(history.data?.items.length||0)<40}>更早的对话</button><button onClick={()=>setOffset(0)} disabled={pending||offset===0}>回到最新</button></div>}
   {history.error&&<ErrorBox error={history.error} onRetry={history.reload}/>}
   {!history.data?.items.length&&!liveQuestion&&<div className="chat-empty"><span className="memory-orb"><LayersMark/></span><h2>{session?'从一个问题，或一份文件开始':'从左侧新建一个对话'}</h2><p>直接聊天，也可以拖入图片、源码或文档。记忆在后台自动整理。</p><div className="suggestions">{questions.map(q=><button key={q} onClick={()=>{setText(q);input.current?.focus()}}>{q}<ArrowRight size={15}/></button>)}</div></div>}
   {history.data?.items.map(record=><article className="chat-record" key={record.id} id={record.id}>
    <div className="question"><span className="message-avatar">你</span><div><p>{record.question}</p><Photos workspaceId={workspace.id} ids={record.media}/>{!!record.attachment_files?.length&&<div className="message-files">{record.attachment_files.map(a=><button key={a.id} onClick={()=>showFile(a.id)}><FileText size={14}/>{a.name}<small>v{a.version}{!a.active?' · 历史版本':''}</small></button>)}</div>}</div></div>
    <div className="assistant"><span className="message-avatar mandol">M</span><div><div className="assistant-label">Mandol <Badge status={record.status}/>{record.route?.generation?.mode&&<small>{({fast:'快速',standard:'标准',deep:'深度'} as Record<string,string>)[record.route.generation.mode]}</small>}</div><Suspense fallback={<p className="answer-text">{record.answer}</p>}><MarkdownAnswer text={record.answer||'尚未生成完整回答，用户消息已保存。'} sources={record.citations} webSources={record.route?.web_sources||[]} onDetail={onDetail}/></Suspense><Evidence sources={record.citations} files={record.route?.attachment_sources||[]} onDetail={onDetail} onFile={showFile}/><WebEvidence route={record.route}/><Suspense fallback={null}><RetrievalExplain route={record.route}/></Suspense>{record.status!=='completed'&&<div className="inline-error">{record.error||'回答尚未完成'}<button onClick={()=>{if(record.status==='stale'){setText(record.question);input.current?.focus()}else ask(record)}} disabled={pending}>{record.status==='stale'?'基于当前来源提问':'重新回答'}</button></div>}</div></div>
   </article>)}
   {liveQuestion&&<article className="chat-record"><div className="question"><span className="message-avatar">你</span><div><p>{liveQuestion}</p></div></div><div className="assistant"><span className="message-avatar mandol">M</span><div><div className="assistant-label">Mandol <small>{status}</small>{pending&&<Spinner/>}</div><StreamText store={stream.current}/><Evidence sources={sources} files={fileSources} onDetail={onDetail} onFile={showFile}/><WebEvidence route={liveRoute}/><Suspense fallback={null}><RetrievalExplain route={liveRoute}/></Suspense></div></div></article>}
  </div>
  <div className="composer-region">
   {receipt&&!pending&&<details className="processing-receipt"><summary><span className="ready-dot"/>消息已保存 · 查看后台处理</summary><p>{receipt.routes?.map((r:string)=>({conversation:'对话知识',engineering:'工程经验候选',visual:'视觉理解',source_code:'源码解析',document:'文档索引'}[r]||r)).join('、')||'基础内容保存与检索索引'} · <a href="#jobs">后台进度</a></p></details>}
   {error&&<ErrorBox error={error} onRetry={failedUpload?()=>upload([failedUpload]):lastSave.current?()=>save(true):lastChat.current?()=>ask(undefined,true):undefined}/>}
   {previewError&&<ErrorBox error={previewError}/>}
   {(queuedUploads>0||failedUpload)&&<p className="composer-note" role="status">{queuedUploads>0?`${queuedUploads} 个附件排队中。`:''}{failedUpload?'上一附件上传失败，可重试。':''}<button onClick={()=>{uploadQueue.current=[];setQueuedUploads(0);setFailedUpload(null);uploadAbort.current?.abort()}}>取消待上传</button></p>}
   <div className="composer" onPaste={paste} onDragOver={e=>e.preventDefault()} onDrop={e=>{e.preventDefault();upload(e.dataTransfer.files)}}>
    <textarea ref={input} aria-label="消息或记忆内容" value={text} onChange={e=>setText(e.target.value)} maxLength={8000} disabled={!session} onKeyDown={e=>{if(e.key==='Enter'&&!e.shiftKey&&!e.nativeEvent.isComposing&&e.nativeEvent.keyCode!==229){e.preventDefault();ask()}}} placeholder="发消息，或粘贴、拖入图片、代码和文档…"/>
    {uploadPreview&&<div className="attachments"><div><img src={uploadPreview.url} alt={'正在上传 '+uploadPreview.name}/><span>{uploadPreview.name} · {uploadProgress}</span><button aria-label="取消图片上传" onClick={()=>uploadAbort.current?.abort()}><X size={13}/></button></div></div>}
    {attachments.length>0&&<div className="attachments">{attachments.map(a=><div key={a.id}>{a.image?<img src={a.url} alt={a.name}/>:<button className="attachment-preview" onClick={()=>showFile(a.id)} aria-label={'预览 '+a.name}><FileText size={24}/></button>}<span>{a.name}{a.version?` · v${a.version}`:''}</span><button aria-label={'移除附件 '+a.name} onClick={()=>{a.url&&URL.revokeObjectURL(a.url);attachmentsRef.current=attachmentsRef.current.filter(v=>v.id!==a.id);setAttachments(attachmentsRef.current)}}><X size={13}/></button></div>)}</div>}
    <div className="composer-toolbar">
     <div className="composer-controls">
      <input type="file" ref={file} aria-label="上传附件" accept="image/png,image/jpeg,image/webp,.py,.ts,.tsx,.js,.jsx,.mjs,.cjs,.rs,.go,.java,.c,.h,.cpp,.hpp,.cc,.md,.txt,.rst,.json,.csv,.yaml,.yml,.toml,.sql,.css" multiple hidden onChange={e=>e.target.files&&upload(e.target.files)}/>
      <button className="composer-add" title={uploadProgress||'添加图片、代码和文档 · 最多 4 个'} aria-label="选择附件" onClick={()=>file.current?.click()} disabled={!session||uploading||pending||attachments.length>=4}>{uploading?<Spinner/>:<Plus size={20}/>}</button>
      <button className={'web-search-toggle '+(webSearch?'active':'')} aria-label="联网搜索" aria-pressed={webSearch} title={capabilities.data?.web.configured?'自动根据问题与近期上下文组织联网查询':'联网工具未配置'} disabled={pending||!capabilities.data?.web.enabled||!capabilities.data?.web.configured} onClick={()=>setWebSearch(v=>!v)}><Globe size={15}/><span className="tool-label">联网搜索</span></button>
      <label className="reasoning-control"><span className="reasoning-caption">推理</span><select aria-label="推理强度" value={reasoning} disabled={pending} onChange={e=>setReasoning(e.target.value)}>{[['fast','快速'],['standard','标准'],['deep','深度']].map(([value,label])=><option value={value} key={value} disabled={!capabilities.data?.generation.modes.includes(value)}>{label}</option>)}</select></label>
     </div>
     <div className="composer-submit">
      <button className="composer-save" aria-label="仅保存" onClick={()=>save()} title="只保存基础内容，不生成回答" disabled={!session||pending||uploading||queuedUploads>0||(!text.trim()&&!attachments.length)}><Save size={15}/><span className="save-label">仅保存</span></button>
      {pending?<button className="primary-button composer-send" aria-label="中断回答" title="中断回答" onClick={()=>abort.current?.abort()} disabled={!abort.current}><Square size={17}/></button>:<button className="primary-button composer-send" aria-label="发送消息" title="发送消息 · Enter" onClick={()=>ask()} disabled={!session||uploading||queuedUploads>0||(!text.trim()&&!attachments.length)}><ArrowUp size={20}/></button>}
     </div>
    </div>
   </div><p className="composer-note">Enter 发送 · Shift + Enter 换行 · 联网时自动组织搜索查询</p>
  </div>
  {preview&&<div className="attachment-modal-backdrop" onClick={()=>setPreview(null)}><section className="attachment-modal" role="dialog" aria-label={'文件预览 '+preview.name} onClick={e=>e.stopPropagation()}><header><strong>{preview.name} · v{preview.version}</strong><button aria-label="关闭文件预览" onClick={()=>setPreview(null)}><X size={18}/></button></header><p>{preview.parser} · {preview.active?'当前有效版本':'历史版本，已退出当前检索'}{preview.active===1&&<button className="danger-button" onClick={()=>withdrawFile()}>停止使用此版本</button>}</p><pre>{preview.text.split('\n').slice(previewOffset,previewOffset+160).map((line,i)=><span className="source-line" key={i}><span>{previewOffset+i+1}</span>{line+'\n'}</span>)}</pre><div className="pagination"><button disabled={previewOffset===0} onClick={()=>setPreviewOffset(v=>Math.max(0,v-160))}>上一段</button><span>{previewOffset+1}–{Math.min(previewOffset+160,preview.text.split('\n').length)} / {preview.text.split('\n').length} 行</span><button disabled={previewOffset+160>=preview.text.split('\n').length} onClick={()=>setPreviewOffset(v=>v+160)}>下一段</button></div></section></div>}
 </div>
}
function LayersMark(){return <svg viewBox="0 0 48 48" width="32" height="32" fill="none" aria-hidden="true"><path d="m24 8 17 9-17 9-17-9 17-9Z" stroke="currentColor" strokeWidth="2.4"/><path d="m7 24 17 9 17-9M7 31l17 9 17-9" stroke="currentColor" strokeWidth="2.4"/></svg>}
