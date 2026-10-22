import {memo,useState} from 'react';

const names:Record<string,string>={raw:'基础证据',profile:'用户画像',episode:'情景 / 时间',entity_relation:'实体关系',engineering:'工程经验',code:'源码结构',ocr:'OCR 文字',visual:'视觉语义',summary:'会话摘要',hybrid:'通用混合',graph:'已存图关系',ocr_lexical:'OCR 词法'};

/** Bounded execution trace only. No candidate texts, private endpoints or keys. */
export default memo(function RetrievalExplain({route}:{route?:Record<string,any>}){
 const [open,setOpen]=useState(false);
 if(!route)return null;
 const trace=route.trace,plan=route.memory_query_plan||trace?.plan;
 return <details className="retrieval-explain" onToggle={e=>setOpen(e.currentTarget.open)}>
  <summary>本次检索详情{route.retrieval_used===false?' · 使用当前上下文':''}</summary>
  {open&&<div>
   {route.generation&&<p>生成模式：{({fast:'快速',standard:'标准',deep:'深度'} as Record<string,string>)[route.generation.mode]||route.generation.mode} · 请求参数 {JSON.stringify(route.generation.parameters||{})}{route.generation.usage?.total_tokens!==undefined?` · ${route.generation.usage.total_tokens} 总 tokens`:''}{route.generation.thinking_observed?' · 已观察到思考输出（不展示内部内容）':''}</p>}
   {route.web?.requested&&<p>联网：{route.web.status} · Search {route.web.calls?.search??'未确认'} 次 / Fetch {route.web.calls?.fetch??'未确认'} 次 · {route.web.timings?.total_ms??'未确认'} ms{route.web.reused_for_same_request?' · 复用同一请求已完成的搜索':''}；网页来源与记忆来源分别验证。</p>}
   {route.web?.plan&&<p>联网规划：{route.web.plan.planner_method} · {route.web.plan.elapsed_ms??0} ms · 额外模型请求 {route.web.plan.model_calls??0} 次{route.web.plan.reused_for_same_request?'（本次复用已完成计划）':''} · {route.web.results?.length??0} 个实际网页结果{route.web.plan.reason_codes?.length?` · ${route.web.plan.reason_codes.join(' / ')}`:''}</p>}
   {route.web?.plan?.primary_query&&<div className="web-query-plan"><strong>实际搜索查询</strong><ol>{[...new Set((route.web.requests||[]).filter((r:any)=>r.kind==='search').map((r:any)=>r.payload.query))].map((q:any)=><li key={q}>{q}</li>)}</ol>{!route.web.calls?.search&&<p>尚无已确认的 Search 调用。</p>}</div>}
   {!!route.web?.warnings?.length&&<p>联网降级：{route.web.warnings.join(' / ')}</p>}
   <p>范围：{route.retrieval_scope==='conversation'?'本对话':route.retrieval_scope==='conversations'||route.retrieval_scope==='workspace'?'当前工作区内已授权对话':'当前授权场景'} · 策略：{trace?.strategy||'既有基线'}{trace?.fast_exact?' · 精确符号快捷访问':''}</p>
   {!trace?<p>本轮没有执行长期检索，使用最近对话、当前附件或模型通用能力。</p>:<>
    {!!plan?.selected_lanes?.length&&<p>规划预算：{plan.selected_lanes.map((lane:any)=>`${names[lane.name]||lane.name}（${lane.candidate_budget}）`).join('、')}{trace.fast_exact?'；本次由唯一精确命中提前完成':''}</p>}
    <p>通道候选：{Object.entries(trace.candidate_counts||{}).map(([k,v])=>`${k} ${v}`).join(' · ')}；RRF 后 {trace.rrf_candidates??0} 条。</p>
    <p>BGE：{trace.reranker_used?`已执行${trace.reranker_candidates!==undefined?`，${trace.reranker_candidates} 条` :''}`:'未执行'} · 失效 / 不适格候选排除 {trace.invalid_excluded??'基线未单独计数'}{trace.expanded_source_unique!==undefined?` · 基础来源 ${trace.expanded_source_unique} 个（去重）`:trace.original_sources!==undefined?` · 来源引用 ${trace.original_sources} 次`:''}</p>
    {!!Object.keys(trace.lanes||{}).length&&<ul>{Object.entries(trace.lanes).map(([key,lane]:[string,any])=><li key={key}>{names[key]||key}：{lane.executed===false?'已跳过':(lane.methods||[]).join(' + ')} · {lane.selected??lane.candidates??0} 条{lane.ms!==undefined?` · ${lane.ms} ms`:''}</li>)}</ul>}
    <p>实际耗时：{Object.entries(trace.timings||{}).filter(([k,v])=>k.endsWith('_ms')&&typeof v==='number').slice(0,12).map(([k,v])=>`${k.replace('_ms','')} ${Number(v).toFixed(1)} ms`).join(' · ')}</p>
    {trace.extra_llm_calls!==undefined&&<p>类型规划额外 LLM 调用：{trace.extra_llm_calls}；对话路由与证据审核另计。</p>}
    {!!trace.fallbacks?.length&&<p>降级 / 兜底：{trace.fallbacks.join('；')}</p>}
    {!!plan?.plan_reason_codes?.length&&<p>决策信号：{plan.plan_reason_codes.join(' · ')}</p>}
   </>}
   {route.context_packing&&<p>送入生成上下文：{route.context_packing.blocks} 个证据块 · {route.context_packing.context_chars} 字符 · 去重 {route.context_packing.duplicate_sources_skipped} 处。回答实际引用见“参考依据”。</p>}
  </div>}
 </details>
});
