import {useState} from 'react';
import {X,Link2,ArrowUpRight,Save,Trash2} from 'lucide-react';
import {api,qs,requestId,useResource} from '../api';
import type {Memory,Workspace} from '../types';
import {Badge,ErrorBox,Spinner,typeNames} from '../ui';
type Relation={relation:string;memory:Memory;direction:string;edge_id:string;category:string};
type History={id:string;items:{version:number;text:string;kind:string;recorded_at?:number}[];current_version:number;complete:boolean;note:string};
export default function Detail({workspace,id,onClose,onSelect,onConversation,refresh=0}:{workspace:Workspace;id:string;onClose:()=>void;onSelect:(id:string)=>void;onConversation?:(id:string)=>void;refresh?:number}){
 const data=useResource<Memory>('/memories/'+encodeURIComponent(id)+'?'+qs({workspace_id:workspace.id,audit:1}),refresh);
 const [editing,setEditing]=useState(false),[text,setText]=useState(''),[error,setError]=useState(''),[busy,setBusy]=useState(false),[relationState,setRelationState]=useState('current'),[historySource,setHistorySource]=useState<string>();
 const relations=useResource<{items:Relation[];truncated:boolean}>('/memories/'+encodeURIComponent(id)+'/relations?'+qs({workspace_id:workspace.id,state_view:relationState}),refresh);
 const history=useResource<History>(historySource?'/memories/'+encodeURIComponent(historySource)+'/history?'+qs({workspace_id:workspace.id}):null);
 const m=data.data;
 async function save(){if(!m||!text.trim())return;setBusy(true);try{await api('/memories/'+id+'?'+qs({workspace_id:workspace.id}),{method:'PUT',body:JSON.stringify({request_id:requestId(),expected_version:m.version,content:text})});setEditing(false);onClose();window.dispatchEvent(new Event('mandol-memory-changed'))}catch(e){setError((e as Error).message)}finally{setBusy(false)}}
 async function remove(){if(!m||!window.confirm('删除此基础记忆后，依赖它的高阶事实会立即失效。继续？'))return;setBusy(true);try{await api('/memories/'+id+'?'+qs({workspace_id:workspace.id,expected_version:m.version,request_id:requestId()}),{method:'DELETE'});onClose();window.dispatchEvent(new Event('mandol-memory-changed'))}catch(e){setError((e as Error).message)}finally{setBusy(false)}}
 return <aside className="detail" aria-label="记忆与来源详情">
  <div className="detail-header"><span><Link2 size={17}/>记忆与来源</span><button aria-label="关闭详情" onClick={onClose}><X size={19}/></button></div>
  {data.error&&<ErrorBox error={data.error} onRetry={data.reload}/>}
  {!m?<div className="loading"><Spinner/></div>:<div className="detail-scroll">
   <div className="detail-tags"><Badge>{m.source_kind==='raw'?'基础':'高阶'}</Badge><Badge>{typeNames[m.memory_type||m.type]||m.memory_type||m.type}</Badge><span>v{m.version}</span></div>
   <h2>{m.source_locator?`${m.source_locator.path}:${m.source_locator.start_line}-${m.source_locator.end_line}`:m.source_kind==='raw'?'基础记录':'来源约束的场景记忆'}</h2>
   <div className={'evidence-state '+(m.eligible?'':'invalid')}>{m.quality?.label||(m.eligible?'来源版本有效':'不符合当前检索资格')} · {m.eligible?'可按相应检索通道使用':'不进入 Agent 检索'}</div>
   {editing?<><textarea className="edit-memory" value={text} onChange={e=>setText(e.target.value)} maxLength={16000} aria-label="修改基础记忆"/><div className="detail-actions"><button className="primary-button" onClick={save} disabled={busy}><Save size={15}/>提交修订</button><button onClick={()=>setEditing(false)}>取消</button></div></>:m.source_locator?<pre className="detail-text">{String(m.raw_data?.code||m.text)}</pre>:<p className="detail-text">{m.text}</p>}
   {m.media_ids?.map(mid=><a key={mid} href={'/api/media/'+mid+'?workspace_id='+encodeURIComponent(workspace.id)} target="_blank" rel="noreferrer"><img className="evidence-image" alt="原图来源" src={'/api/media/'+mid+'?workspace_id='+encodeURIComponent(workspace.id)+'&thumb=true'} loading="lazy"/></a>)}
   {m.source_kind==='raw'&&m.eligible&&!m.source_locator&&!editing&&<div className="detail-actions"><button className="secondary-button" onClick={()=>{setText(m.text);setEditing(true)}}>修订基础记录</button><button className="danger-button" aria-label="删除基础记忆" onClick={remove} disabled={busy}><Trash2 size={16}/></button></div>}
   {error&&<ErrorBox error={error}/>}<div className="detail-divider"/>
   {(m.session_id||m.sources?.[0]?.session_id)&&<button className="secondary-button" onClick={()=>onConversation?.(m.session_id||m.sources![0].session_id)}>返回来源对话</button>}
   <h3>基础证据 <small>{m.sources?.length||0}</small></h3>
   {m.source_kind==='raw'?<p className="muted">这是基础输入；关联高阶结果可在下方双向关系中查看。</p>:!m.sources?.length?<p className="muted">没有适格的基础来源，不能作为已验证事实使用。</p>:m.sources.map(source=><article className="source-card" key={source.id}>
    <div><span>{source.source_locator?`${source.source_locator.path}:${source.source_locator.start_line}-${source.source_locator.end_line}`:source.role==='user'?'用户记录':'基础记录'} · 当前 v{source.version}</span><button aria-label="查看基础来源" onClick={()=>onSelect(source.id)}><ArrowUpRight size={16}/></button></div>
    {source.version!==source.expected_version&&<Badge status="failed">依赖 v{source.expected_version}，当前已变化</Badge>}
    <p>{source.text}</p>{source.quotes?.map((quote,i)=><blockquote key={i}>{quote}</blockquote>)}
    {source.version!==source.expected_version&&<button className="muted-button" onClick={()=>setHistorySource(source.id)}>查看已保存的来源版本</button>}
    {source.media_ids?.map(mid=><img key={mid} alt="原图证据" src={'/api/media/'+mid+'?workspace_id='+encodeURIComponent(workspace.id)+'&thumb=true'} loading="lazy"/>)}</article>)}
   <div className="detail-divider"/><h3>{m.source_kind==='raw'?'由此构建的高阶记忆':'双向来源与语义关系'}</h3>
   <select aria-label="详情关系状态" value={relationState} onChange={e=>setRelationState(e.target.value)}><option value="current">当前有效关联</option><option value="all">含历史失效关联</option></select>
   {relations.loading&&!relations.data?<Spinner/>:relations.data?.items.length?relations.data.items.map((r,i)=><article className="source-card" key={r.edge_id+':'+r.direction+':'+i}><div><Badge>{r.category==='provenance'?'来源依赖':'语义/结构'}</Badge><button aria-label="查看关联记忆" onClick={()=>onSelect(r.memory.id)}><ArrowUpRight size={16}/></button></div><p>{r.memory.text.slice(0,220)}</p><small>{r.relation} · {r.direction==='incoming'?'指向此记录':'从此记录出发'} · {r.memory.quality?.label||r.memory.lifecycle}</small></article>):<p className="muted">当前筛选下没有已保存的关联。</p>}
   {relations.error&&<ErrorBox error={relations.error} onRetry={relations.reload}/>}{relations.data?.truncated&&<p className="muted">关联超出单次预算，请在关系图聚焦和分页查看。</p>}
   <div className="detail-divider"/>
   <button className="secondary-button" onClick={()=>setHistorySource(historySource?undefined:m.id)}>查看已保存版本与摄入记录</button>
   {historySource&&<section className="version-history" aria-label="已保存版本"><h3>版本与输入审计</h3>{history.loading&&!history.data?<Spinner/>:history.data?.items.map(v=><article className="source-card" key={v.version}><strong>v{v.version} · {v.kind==='canonical_current'?'当前 canonical 记录':'持久化摄入内容'}</strong><p>{v.text}</p></article>)}{history.error&&<ErrorBox error={history.error} onRetry={history.reload}/>}<p className="muted">仅显示实际保留的记录。{history.data?.complete===false?'没有完整旧版本快照，未补造缺失内容。':'摄入内容不等于完整历史 canonical 元数据快照。'}</p></section>}
   {m.code_associations?.length?<><div className="detail-divider"/><h3>关联源码 · 原文路径提及</h3>{m.code_associations.map(link=><article className="source-card" key={link.memory_id}><div><span>{link.path}</span><button aria-label="查看关联源码" disabled={!link.current_reference_valid} onClick={()=>onSelect(link.memory_id)}><ArrowUpRight size={16}/></button></div><p>{link.current_reference_valid?'当前文件引用有效；工程结论仍由历史记录支持':'当前代码引用已失效，历史工程证据保留'}</p></article>)}</>:null}
   <div className="detail-divider"/><dl className="detail-dl"><dt>存储层级</dt><dd>{m.actual_level}</dd><dt>验证级别</dt><dd>{m.verification}</dd><dt>资格</dt><dd>{m.eligible?'当前来源适格':'历史保留，不供当前检索'}</dd><dt>索引投影</dt><dd>{m.dense_ready?'所属范围向量发布一致':'所属范围向量待发布'} · {m.lexical_ready?'词法已发布':'词法未发布'}</dd></dl>
   <details className="audit"><summary>来源审计信息</summary><pre>{JSON.stringify({id:m.id,version:m.version,quality:m.quality,generation:m.generation,metadata:m.metadata},null,2)}</pre></details>
  </div>}
 </aside>;
}
