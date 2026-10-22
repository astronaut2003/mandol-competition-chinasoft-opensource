import type {ReactNode} from 'react';
import {LoaderCircle,AlertCircle,ArrowUpRight,Database} from 'lucide-react';
import type {Memory} from './types';
export const names:Record<string,string>={textual:'长期对话',code:'工程记忆',multimodal:'视觉记忆',conversation:'长期对话',repository:'仓库扫描与解析',source_unavailable:'源码来源不可用',lexical_ready:'词法就绪 · 向量待发布',ready:'可检索',scanning:'扫描与解析中',ingest:'基础记忆保存',index:'检索索引',derive:'高阶记忆构建',pending:'待处理',blocked:'等待基础入库',running:'处理中',retry_wait:'等待重试',completed:'已完成',failed:'失败',cancelled:'已取消',interrupted:'已中断',stale:'来源已变化'};
export const typeNames:Record<string,string>={observation:'基础记忆',engineering_raw:'工程基础记录',hierarchical:'会话摘要',entity:'实体',relation:'关系',episode:'事件',profile:'用户画像',engineering_experience:'工程经验',engineering_knowledge:'工程知识',visual_memory:'视觉理解',visual_episode:'视觉会话窗口'};
names.cleared='记忆已清除 · 可重新同步';
Object.assign(names,{conversation_derivation:'对话情景与画像构建',engineering_experience_derivation:'工程经验构建',visual_derivation:'视觉语义构建',repository_scan_parse:'仓库扫描与解析',raw_ingestion:'基础记忆保存',retrieval_index:'检索投影'});
Object.assign(typeNames,{source_attachment:'基础文件',conversation_message:'来源对话',conversation_summary:'会话摘要',episodic_event:'情景事件',user_profile:'User Profile',engineering_record:'工程基础记录',source_chunk:'源码片段',code_symbol:'代码符号',code_dependency:'导入与依赖',repository_map:'Repository Map',image_text:'原图及说明',visual_description:'视觉描述',ocr_text:'OCR 文字',visual_object:'可见对象',visual_ocr:'OCR 文字'});
export const families:Record<string,string>={textual:'conversation',code:'code',multimodal:'multimodal'};
export function memoryLabel(memory:Memory){return typeNames[memory.memory_type||memory.type]||memory.memory_type||memory.type}
export function Spinner(){return <LoaderCircle className="spin" size={19}/>}
export function ErrorBox({error,onRetry}:{error:string;onRetry?:()=>void}){return <div className="error-box" role="alert"><AlertCircle size={18}/><span>{memoryTerms(error)}</span>{onRetry&&<button onClick={onRetry}>重试</button>}</div>}
export function Empty({title,children}:{title:string;children?:ReactNode}){return <div className="empty"><Database size={30}/><h3>{title}</h3><p>{children}</p></div>}
export function Badge({status,children}:{status?:string;children?:ReactNode}){return <span className={'badge '+(status||'')}>{children||names[status||'']||status}</span>}
export function MemoryCard({memory,onDetail}:{memory:Memory;onDetail:(id:string)=>void}){return <button className="memory-card" onClick={()=>onDetail(memory.id)}><div className="card-meta"><Badge>{memory.source_kind==='derived'?'高阶':'基础'}</Badge><span>{memoryLabel(memory)}</span><ArrowUpRight size={15}/></div><p>{memory.text}</p>{memory.media_ids?.length>0&&<img className="card-image" alt="原图证据" src={`/api/media/${memory.media_ids[0]}?workspace_id=${encodeURIComponent(memory.workspace_id)}&thumb=true`} loading="lazy"/>}<div className="card-foot"><span>{memory.eligible?(memory.dense_ready?'可检索 · 向量已发布':'可检索 · 词法就绪'):'来源已变化 · 已失效'}</span><span>v{memory.version}</span></div></button>}

export function memoryTypeColor(type:string,kind?:string){if(kind==='raw')return '#238f6b';return ({user_profile:'#7157bc',episodic_event:'#bd8236',entity:'#5265d9',relation:'#5265d9',code_symbol:'#307b99',code_dependency:'#307b99',visual_description:'#a5688f',ocr_text:'#a5688f',visual_object:'#a5688f'} as Record<string,string>)[type]||'#5265d9'}

export function memoryTerms(text:string){return text.replace(/原始记忆/g,'基础记忆').replace(/派生记忆/g,'高阶记忆').replace(/原始与派生/g,'基础与高阶').replace(/原始保存/g,'基础保存').replace(/派生构建/g,'高阶构建')}
