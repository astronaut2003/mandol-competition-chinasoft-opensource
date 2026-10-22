import {useEffect,useRef,useState} from 'react';
import cytoscape,{type Core} from 'cytoscape';
import {Search,ArrowLeft,Layers,Network,Link2} from 'lucide-react';
import {api,qs,useResource} from '../api';
import type {Memory,PageProps,Paged} from '../types';
import type {SemanticSchema} from './MemorySchemaOverview';
import {Badge,Empty,ErrorBox,Spinner,typeNames,memoryTypeColor} from '../ui';
import {attachLabelAvoidance,safeFit,shortLines,sourceLayout,type Point} from '../graph/layout';
type Node={id:string;label:string;short:string;title:string;memory:Memory;is_group:boolean;record_count:number;member_ids:string[]};
type Edge={id:string;source:string;target:string;relation:string;category:string;record_count:number;edge_ids:string[]};
type Graph={nodes:Node[];edges:Edge[];total:number;record_total:number;visible_record_count:number;context_nodes_included:number;page_size:number;truncated:boolean;hidden_relations:number;edge_budget_reached?:boolean;selection?:{complete:boolean}};
type Props={scope:string;kind:string;memoryType:string;initialFocus?:string;stateView?:string}&PageProps;
const visibleNodes=(base:Graph|null,extra:Graph|null,budget:number)=>Array.from(new Map([...(extra?.nodes||[]),...(base?.nodes||[])].map(n=>[n.id,n])).values()).slice(0,budget);

export default function KnowledgeGraph({workspace,session,jobTick,onDetail,onNavigateSession,scope,kind,memoryType,initialFocus,stateView='current'}:Props){
 const [mode,setMode]=useState(initialFocus?'provenance':'semantic'),[group,setGroup]=useState<string|null>(memoryType||null),[focus,setFocus]=useState<string|undefined>(initialFocus);
 const [budget,setBudget]=useState(10),[offset,setOffset]=useState(0),[selected,setSelected]=useState<Node|null>(null),[extra,setExtra]=useState<Graph|null>(null),[expanding,setExpanding]=useState(false);
 const [error,setError]=useState(''),[input,setInput]=useState(''),[query,setQuery]=useState(''),[relation,setRelation]=useState(''),[hover,setHover]=useState(''),[aggregate,setAggregate]=useState(true),[expandedGroup,setExpandedGroup]=useState<string>(),[memberPage,setMemberPage]=useState(0);
 const host=useRef<HTMLDivElement>(null),instance=useRef<Core|null>(null),positions=useRef(new Map<string,Point>()),pinned=useRef(new Set<string>()),expandAbort=useRef<AbortController|null>(null),needsFit=useRef(true);
 const labels=useRef<ReturnType<typeof attachLabelAvoidance>|null>(null),callbacks=useRef({onDetail});callbacks.current={onDetail};
 const scopeParams={workspace_id:workspace.id,session_id:scope==='conversation'?session?.id:undefined,retrieval_scope:scope,state_view:stateView};
 const schema=useResource<SemanticSchema>(session||scope==='conversations'?'/memory-schema?'+qs(scopeParams):null,jobTick);
 const baseParams={...scopeParams,source_kind:kind,relationship_view:mode,limit:budget,aggregate:aggregate?'true':'false'};
 const path=focus||group?'/memory-graph?'+qs({...baseParams,memory_type:focus?undefined:group||undefined,center:focus,offset:focus?0:offset,expand_group:expandedGroup}):null;
 const data=useResource<Graph>(path,jobTick);
 const found=useResource<Paged<Memory>>(query?'/memories?'+qs({...scopeParams,q:query,limit:8}):null,jobTick,'/memories?'+qs(scopeParams));
 const members=useResource<Paged<Memory>>(selected?.is_group?'/memory-groups/'+encodeURIComponent(selected.id)+'?'+qs({...scopeParams,offset:memberPage,limit:12}):null,jobTick,selected?.is_group?'/memory-groups/'+encodeURIComponent(selected.id)+'?'+qs(scopeParams):null);
 useEffect(()=>{const t=setTimeout(()=>setQuery(input.trim()),260);return()=>clearTimeout(t)},[input]);
 useEffect(()=>{expandAbort.current?.abort();setExtra(null);setSelected(null);positions.current.clear();pinned.current.clear();setGroup(memoryType||null);setFocus(initialFocus);setOffset(0);needsFit.current=true;if(initialFocus)setMode('provenance')},[workspace.id,session?.id,scope,kind,memoryType,initialFocus,stateView]);
 useEffect(()=>{expandAbort.current?.abort();setExtra(null);setSelected(null);setMemberPage(0);setError('');needsFit.current=true},[path]);
 useEffect(()=>{expandAbort.current?.abort();setExtra(null)},[jobTick]);
 useEffect(()=>{
  if(!host.current)return;const element=host.current;
  const cy=cytoscape({container:element,elements:[],layout:{name:'preset'},minZoom:.04,maxZoom:2.5,wheelSensitivity:.16,hideEdgesOnViewport:true,textureOnViewport:true,style:[
   {selector:'node',style:{'background-color':'data(color)','width':32,'height':32,'label':'data(short)','font-size':12,'color':'#5d6980','text-valign':'bottom','text-margin-y':8,'text-wrap':'wrap','text-max-width':'120px','text-background-opacity':1,'text-background-color':'#ffffff','text-background-padding':'3px','text-background-shape':'roundrectangle','border-width':2,'border-color':'#ffffff','text-events':'yes'}},
   {selector:'node[valid="invalid"]',style:{'border-color':'#bc4454','border-style':'dashed'}},
   {selector:'node[isGroup="yes"]',style:{'shape':'round-rectangle','width':48,'height':38,'border-width':3}},
   {selector:'edge',style:{'width':1.2,'opacity':.32,'line-color':'#9ba7be','target-arrow-color':'#9ba7be','target-arrow-shape':'triangle','curve-style':'bezier','arrow-scale':.7}},
   {selector:'edge[category="provenance"]',style:{'line-style':'dashed','curve-style':'taxi','taxi-direction':'leftward','taxi-turn':'50%'}},
   {selector:'.source-path',style:{'opacity':1,'line-color':'#5265d9','target-arrow-color':'#5265d9','width':2.5}},
   {selector:'edge.dimmed',style:{'opacity':.055}},
   {selector:'node:selected',style:{'border-width':4,'border-color':'#aab7fb'}},
   {selector:'.label-hidden',style:{'label':''}}
  ]});instance.current=cy;labels.current=attachLabelAvoidance(cy,element);
  const theme=()=>{const style=getComputedStyle(element),surface=style.getPropertyValue('--surface').trim(),muted=style.getPropertyValue('--text').trim();cy.style().selector('node').style({'text-background-color':surface||'#fff','color':muted||'#5d6980'}).update();labels.current?.schedule()};theme();
  const observer=new MutationObserver(theme);observer.observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
  const choose=(n:cytoscape.NodeSingular)=>{
   const node=n.data('record') as Node;setSelected(node);setMemberPage(0);
   if(!node.is_group)callbacks.current.onDetail(node.memory.id);
   cy.elements().removeClass('source-path dimmed neighbor-focus');cy.nodes().unselect();n.select();
   const local=n.closedNeighborhood();cy.edges().difference(local.edges()).addClass('dimmed');local.edges().addClass('source-path');local.nodes().addClass('neighbor-focus');labels.current?.schedule();
  };
  cy.on('tap','node',e=>choose(e.target));
  cy.on('dragfree','node',e=>{positions.current.set(e.target.id(),e.target.position());pinned.current.add(e.target.id())});
  cy.on('mouseover','node',e=>{e.target.addClass('hot');setHover(e.target.data('title'))});cy.on('mouseout','node',e=>{e.target.removeClass('hot');setHover('')});
  cy.on('mouseover','edge',e=>setHover(`${e.target.data('relation')} · ${e.target.data('category')==='provenance'?'来源依赖':'已存语义/结构关系'} · ${e.target.data('record_count')||1} 条实际关系`));cy.on('mouseout','edge',()=>setHover(''));
  let oldWidth=element.clientWidth,oldHeight=element.clientHeight;let resizeTimer:ReturnType<typeof setTimeout>;
  const resize=new ResizeObserver(()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>{
   const start=performance.now(),node=cy.nodes(':selected').first(),point=node.length?node.renderedPosition():null;
   const width=element.clientWidth,height=element.clientHeight;cy.resize();
   if(point&&oldWidth&&oldHeight&&width&&height){const next=node.renderedPosition(),target={x:Math.max(72,Math.min(width-72,point.x*width/oldWidth)),y:Math.max(60,Math.min(height-60,point.y*height/oldHeight))};cy.panBy({x:target.x-next.x,y:target.y-next.y})}
   oldWidth=width;oldHeight=height;labels.current?.schedule();element.dataset.resizeMs=(performance.now()-start).toFixed(3);
  },60)});resize.observe(element);
  return()=>{clearTimeout(resizeTimer);expandAbort.current?.abort();resize.disconnect();observer.disconnect();labels.current?.destroy();labels.current=null;cy.destroy();instance.current=null};
 },[workspace.id]);
 useEffect(()=>{
  const cy=instance.current,base=data.data;if(!cy)return;
  if(!base){cy.elements().remove();host.current?.setAttribute('data-node-count','0');return}
  // A requested neighborhood gets the finite display budget before unrelated
  // page candidates. Otherwise a full page silently starves every new node.
  const nodes=visibleNodes(base,extra,budget),ids=new Set(nodes.map(n=>n.id));
  const edges=Array.from(new Map([...base.edges,...(extra?.edges||[])].map(e=>[e.id,e])).values()).filter(e=>ids.has(e.source)&&ids.has(e.target)&&(!relation||e.relation===relation));
  const start=performance.now(),empty=cy.nodes().empty();let added=false;
  cy.batch(()=>{
   cy.edges().filter(e=>!edges.some(x=>x.id===e.id())).remove();cy.nodes().filter(n=>!ids.has(n.id())).remove();
   nodes.forEach((n,i)=>{
    const meta={id:n.id,short:shortLines(n.is_group?(n.short||n.label).slice(0,10)+' · '+n.record_count+'条':n.short||n.label),title:n.title||n.label,kind:n.memory.source_kind,valid:n.memory.eligible?'current':'invalid',isGroup:n.is_group?'yes':'no',color:memoryTypeColor(n.memory.memory_type,n.memory.source_kind),memory:n.memory,record:n};
    const existing=cy.getElementById(n.id);if(existing.length){existing.data(meta);if(focus===n.memory.id)existing.addClass('search-hit');return}
    const cached=positions.current.get(n.id),anchor=cy.nodes(':selected').first(),center=anchor.length?anchor.position():{x:0,y:0},angle=i*2.39996;
    const position=cached||(extra?{x:center.x+180*Math.cos(angle),y:center.y+180*Math.sin(angle)}:{x:(i%5)*200,y:Math.floor(i/5)*125});
    const created=cy.add({data:meta,position});if(focus===n.memory.id)created.addClass('search-hit');added=true;
   });
   edges.forEach(e=>{if(!cy.getElementById(e.id).length)cy.add({data:e})});
  });
  const layoutStart=performance.now();
  if(added&&!extra&&mode==='provenance')sourceLayout(cy,pinned.current);
  else if(added&&!extra&&mode==='semantic'&&empty&&nodes.length>1){cy.layout({name:'cose',animate:false,randomize:false,fit:false,nodeDimensionsIncludeLabels:true,numIter:350,nodeRepulsion:()=>18000,nodeOverlap:12,idealEdgeLength:()=>170,componentSpacing:90,gravity:.2}).run()}
  cy.nodes().forEach(n=>{positions.current.set(n.id(),n.position())});while(positions.current.size>1000){const key=positions.current.keys().next().value;if(key===undefined)break;positions.current.delete(key)}
  // A grouped overview keeps the canvas display:none. Refresh Cytoscape's
  // cached dimensions before the first fit after revealing it; ResizeObserver
  // deliberately preserves the viewport for subsequent drawer changes.
  if(needsFit.current&&!extra){cy.resize();safeFit(cy);needsFit.current=false}
  if(selected&&!ids.has(selected.id))setSelected(null);
  labels.current?.schedule();
  if(host.current){host.current.dataset.nodeCount=String(nodes.length);host.current.dataset.edgeCount=String(edges.length);host.current.dataset.layoutMs=(performance.now()-layoutStart).toFixed(3);host.current.dataset.renderMs=(performance.now()-start).toFixed(3)}
 },[data.data,extra,budget,relation,mode]);
 async function expand(){if(!selected||selected.is_group)return;expandAbort.current?.abort();const ctrl=new AbortController();expandAbort.current=ctrl;setExpanding(true);setError('');try{const next=await api<Graph>('/memory-graph?'+qs({...baseParams,source_kind:undefined,memory_type:undefined,center:selected.memory.id,offset:0}),{signal:ctrl.signal});if(!ctrl.signal.aborted)setExtra(current=>({ ...next,nodes:Array.from(new Map([...(current?.nodes||[]),...next.nodes].map(n=>[n.id,n])).values()).slice(0,80),edges:Array.from(new Map([...(current?.edges||[]),...next.edges].map(e=>[e.id,e])).values()).slice(0,256)}))}catch(e){if(!ctrl.signal.aborted)setError((e as Error).message)}finally{if(!ctrl.signal.aborted)setExpanding(false)}}
 function overview(){expandAbort.current?.abort();setGroup(null);setFocus(undefined);setSelected(null);setExtra(null);setExpandedGroup(undefined);setOffset(0);setInput('');setQuery('');needsFit.current=true}
 function openNode(m:Memory){setMode(m.source_kind==='raw'?'provenance':mode);setFocus(m.id);setGroup(null);setInput('');setQuery('');setOffset(0);needsFit.current=true}
 const groups=(schema.data?.types||[]).filter(t=>t.count>0&&(mode==='provenance'||t.source_kind==='derived')&&(!kind||t.source_kind===kind)&&(!memoryType||t.id===memoryType));
 const actualNodes=visibleNodes(data.data,extra,budget);
 const actualIds=new Set(actualNodes.map(n=>n.id)),allEdges=Array.from(new Map([...(data.data?.edges||[]),...(extra?.edges||[])].map(e=>[e.id,e])).values()).filter(e=>actualIds.has(e.source)&&actualIds.has(e.target));
 return <div className="knowledge-view">
  <div className="graph-mode-tabs"><button className={mode==='semantic'?'active':''} onClick={()=>{setMode('semantic');overview()}}><Network size={16}/>语义关系</button><button className={mode==='provenance'?'active':''} onClick={()=>{setMode('provenance');overview()}}><Link2 size={16}/>来源关系</button></div>
  <p className="muted">{stateView==='current'?'当前有效知识 · 与 Agent 来源资格一致，索引就绪单独显示。':'历史审计 · 展示已保存的失效记录及必要来源，不重新赋予检索资格。'}{mode==='provenance'?' 左：基础来源；右：高阶知识。':' 来源依赖默认隐藏；同名不代表同一实体。'}</p>
  <div className="search-input"><Search size={17}/><input aria-label="搜索图节点" placeholder="搜索当前授权范围的节点" value={input} onChange={e=>setInput(e.target.value)}/></div>
  {query&&<div className="graph-search-results" aria-busy={found.loading||input.trim()!==query}>{found.loading&&!found.data?<Spinner/>:found.data?.items.length?found.data.items.map(m=><button key={m.id} onClick={()=>openNode(m)}><Badge>{typeNames[m.memory_type]||m.memory_type}</Badge>{m.text.slice(0,65)}</button>):<p>没有匹配节点。</p>}</div>}
  {(error||data.error||schema.error)&&<ErrorBox error={error||data.error||schema.error} onRetry={()=>{data.reload();schema.reload()}}/>}
  {!group&&!focus?<>
   <div className="list-heading"><span>按类型浏览 · 当前视图 {schema.data?.instance.counted||0} 条记录</span>{schema.loading&&<Spinner/>}</div>
   {groups.length?<div className="graph-groups">{groups.map(g=><button className="graph-group" key={g.id} style={{borderLeftColor:memoryTypeColor(g.id,g.source_kind)}} onClick={()=>{setGroup(g.id);setOffset(0)}}><Layers size={21}/><strong>{g.name}</strong><span>{g.count} 条记录 · {g.valid} 当前有效</span><small>点击加载局部知识</small></button>)}</div>:<Empty title={stateView==='current'?'当前范围还没有适格知识':'当前范围没有已保存的失效知识'}>构建与来源状态见记忆模式；基础交互历史保留在时间线。</Empty>}
   <p className="graph-aggregation-note">分组和等价折叠仅用于显示，实际 UID、版本、主体和来源保持独立。</p>
  </>:<div className="filters graph-controls">
   <button className="secondary-button" onClick={overview}><ArrowLeft size={14}/>返回概览</button>
   <label>可见节点 <select aria-label="图节点预算" value={budget} onChange={e=>{setBudget(Number(e.target.value));setOffset(0);needsFit.current=true}}>{[10,24,48,80].map(n=><option key={n} value={n}>{n}</option>)}</select></label>
   <label><input type="checkbox" checked={aggregate} onChange={e=>{setAggregate(e.target.checked);setExpandedGroup(undefined)}}/>折叠等价记录</label>
   <select aria-label="关系类型" value={relation} onChange={e=>setRelation(e.target.value)}><option value="">当前局部的全部关系</option>{Array.from(new Set(allEdges.map(e=>e.relation))).map(r=><option key={r} value={r}>{r}</option>)}</select>
   <button className="secondary-button" onClick={()=>instance.current&&safeFit(instance.current)}>适应窗口</button>
   <button className="secondary-button" onClick={expand} disabled={!selected||selected.is_group||expanding}>{expanding?<Spinner/>:'展开选中节点一跳'}</button>
   <button className="secondary-button" disabled={!selected} onClick={()=>{const cy=instance.current;if(cy&&selected)safeFit(cy,cy.getElementById(selected.id).closedNeighborhood())}}>聚焦选中关系</button>
   <button className="muted-button" disabled={!extra&&!expandedGroup} onClick={()=>{setExtra(null);setExpandedGroup(undefined)}}>收起展开</button>
  </div>}
  <div className={'knowledge-canvas '+(!group&&!focus?'graph-canvas-hidden':'')} role="img" aria-label="真实记忆来源关系图" ref={host}/>
  {(group||focus)&&<>
   <div className="graph-legend"><span><i style={{background:'#238f6b'}}/>基础记忆</span><span><i style={{background:'#5265d9'}}/>高阶记忆</span><span>方形：等价记录组</span><span>虚线：来源依赖</span><span>虚线边框：历史失效</span></div>
   <div className="list-heading"><span>{actualNodes.length} 个显示节点 · {data.data?.visible_record_count||0} 条实际记录（含 {data.data?.context_nodes_included||0} 条关联上下文） · {allEdges.filter(e=>!relation||e.relation===relation).length} 组关系</span>{data.loading&&<Spinner/>}</div>
   {(data.data?.edge_budget_reached||data.data?.selection?.complete===false)&&<p className="muted">当前结果触及查询预算，未加载全库；请聚焦节点、筛选或分页。</p>}
   <p className="graph-hover" aria-live="polite">{hover||'短标签自动避让；完整标题可 Hover，完整内容可从节点列表或详情访问。'}</p>
   {selected&&<div className="graph-selection"><Badge>{selected.is_group?`${selected.record_count} 条等价记录`:typeNames[selected.memory.memory_type]||selected.memory.memory_type}</Badge><span>{selected.title} · {selected.memory.eligible?'当前有效':'历史失效'}</span>{selected.is_group?<button onClick={()=>{setExpandedGroup(selected.id);setSelected(null);needsFit.current=true}}>展开实际 UID</button>:<><button onClick={()=>onDetail(selected.memory.id)}>查看记忆详情</button>{selected.memory.session_id&&<button onClick={()=>onNavigateSession?.(selected.memory.session_id)}>返回来源对话</button>}</>}</div>}
   {selected?.is_group&&<section className="group-inspector" aria-label="等价记录成员"><h3>显示折叠 · {members.data?.total||selected.record_count} 条真实记忆</h3><p className="muted">记录没有永久合并；每条来源、时间和版本独立。</p>{members.loading&&!members.data?<Spinner/>:members.data?.items.map(m=><article key={m.id}><button onClick={()=>onDetail(m.id)}>{m.text.slice(0,100)}<small>{m.id.slice(0,24)}… · v{m.version} · {m.sources?.length||0} 条来源</small></button></article>)}{members.error&&<ErrorBox error={members.error} onRetry={members.reload}/>}<div className="pagination"><button disabled={!memberPage||members.loading} onClick={()=>setMemberPage(x=>Math.max(0,x-12))}>上一页成员</button><button disabled={members.loading||memberPage+12>=(members.data?.total||0)} onClick={()=>setMemberPage(x=>x+12)}>下一页成员</button></div></section>}
   <details className="graph-node-list"><summary>当前节点列表 · 隐藏标签仍可访问</summary>{actualNodes.map(n=><button key={n.id} title={n.title} onFocus={()=>{instance.current?.getElementById(n.id).addClass('hot');labels.current?.schedule()}} onBlur={()=>{instance.current?.getElementById(n.id).removeClass('hot');labels.current?.schedule()}} onClick={()=>{const node=instance.current?.getElementById(n.id);if(node?.length)node.emit('tap')}}>{n.title}{n.is_group&&` · ${n.record_count} 条`}</button>)}</details>
   {!focus&&<div className="pagination"><button disabled={!offset} onClick={()=>{setOffset(x=>Math.max(0,x-(data.data?.page_size||budget)));needsFit.current=true}}>上一页</button><span>第 {Math.floor(offset/(data.data?.page_size||budget))+1} 页 · {data.data?.total||0} 个显示对象 / {data.data?.record_total||0} 条记录</span><button disabled={offset+(data.data?.page_size||budget)>=(data.data?.total||0)} onClick={()=>{setOffset(x=>x+(data.data?.page_size||budget));needsFit.current=true}}>下一页</button></div>}
  </>}
 </div>;
}
