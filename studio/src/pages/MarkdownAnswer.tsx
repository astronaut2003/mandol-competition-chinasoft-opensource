import {memo,useState} from 'react';
import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type {Memory,WebSource} from '../types';

function CodeBlock({children,node:_node,...props}:React.ComponentProps<'pre'>&{node?:unknown}){
 const [copied,setCopied]=useState(false);
 return <div className="code-block"><button aria-label="复制代码" onClick={async e=>{
  const text=e.currentTarget.parentElement?.querySelector('pre')?.textContent||'';
  try{await navigator.clipboard.writeText(text);setCopied(true);setTimeout(()=>setCopied(false),1800)}catch{setCopied(false)}
 }}>{copied?'已复制':'复制'}</button><pre {...props}>{children}</pre></div>
}
export default memo(function MarkdownAnswer({text,sources,onDetail,webSources=[]}:{text:string;sources:(Memory & {citation_index?:number})[];onDetail:(id:string)=>void;webSources?:WebSource[]}){
 const linked=text.split(/(```[\s\S]*?```|`[^`]*`)/g).map((part,i)=>i%2?part:part.replace(/\[(\d+)\](?!\()/g,(original,n)=>sources.some((s,i)=>(s.citation_index||i+1)===Number(n))?`[${n}](#evidence-${n})`:original).replace(/\[(W\d+)\](?:\([^\n)]*\))?/g,(_original,id)=>webSources.some(s=>s.id===id)?`[${id}](#web-${id})`:`[${id}]`)).join('');
 return <div className="answer-text markdown-answer"><Markdown remarkPlugins={[remarkGfm]} skipHtml components={{
  pre:CodeBlock,
  img:({alt})=><span>{alt?`[图片：${alt}]`:'[图片链接]'}</span>,
  a:({href,children})=>{const match=href?.match(/^#evidence-(\d+)$/);const memory=match&&sources.find((s,i)=>(s.citation_index||i+1)===Number(match[1]));const webId=href?.match(/^#web-(W\d+)$/)?.[1],web=webSources.find(s=>s.id===webId);return memory?<button className="citation" onClick={()=>onDetail(memory.id)}>{children}</button>:<a className={web?'web-citation':undefined} title={web?.title} href={web?.url||href} target="_blank" rel="noreferrer noopener">{children}</a>}
 }}>{linked}</Markdown></div>
});
