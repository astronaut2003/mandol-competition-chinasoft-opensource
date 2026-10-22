import type {Core,NodeSingular} from 'cytoscape';
export type Point={x:number;y:number};
type Box={x1:number;y1:number;x2:number;y2:number};
const overlap=(a:Box,b:Box,pad=3)=>a.x1<b.x2+pad&&a.x2>b.x1-pad&&a.y1<b.y2+pad&&a.y2>b.y1-pad;

/** Two-line display text, independent of body content and selection. */
export function shortLines(value:string){
 const lines:string[]=[];let line='',width=0;
 for(const c of value){const size=/[^\x00-\xff]/.test(c)?2:1;if(width+size>16){lines.push(line);line='';width=0}if(lines.length===2)break;line+=c;width+=size}
 if(lines.length<2&&line)lines.push(line);
 if(lines.join('').length<value.length)lines[lines.length-1]=lines.at(-1)!.replace(/.$/,'…');
 return lines.slice(0,2).join('\n');
}

export function safeFit(cy:Core,nodes=cy.elements()){
 if(!nodes.length)return;const box=nodes.boundingBox({includeLabels:false,includeOverlays:false,includeUnderlays:false});const z=Math.min(1.2,Math.max(.04,Math.min((cy.width()-144)/Math.max(1,box.w),(cy.height()-144)/Math.max(1,box.h))));cy.zoom(z);cy.pan({x:cy.width()/2-(box.x1+box.x2)*z/2,y:cy.height()/2-(box.y1+box.y2)*z/2})
}

export function sourceLayout(cy:Core,pinned:Set<string>){
 const basic=cy.nodes().filter(n=>n.data('kind')==='raw').sort((a,b)=>a.id().localeCompare(b.id()));
 const higher=cy.nodes().filter(n=>n.data('kind')!=='raw');
 const rank=new Map(basic.map((n,i)=>[n.id(),i]));
 const score=(n:NodeSingular)=>{const ids=n.neighborhood('node').map(x=>rank.get(x.id())).filter((x):x is number=>x!==undefined);return ids.length?ids.reduce((a,b)=>a+b,0)/ids.length:1e6};
 const sorted=higher.sort((a,b)=>score(a as NodeSingular)-score(b as NodeSingular)||a.id().localeCompare(b.id()));
 // Size-aware lane spacing; barycentric order reduces crossings. Stored edges
 // are never removed. Pinned/manual positions are left unchanged.
 for(const [column,nodes] of [[0,basic],[1,sorted]] as const){let y=48;
  nodes.forEach(n=>{const lineCount=String(n.data('short')).split('\n').length;const height=Math.max(86,48+lineCount*18+n.degree()*2);if(!pinned.has(n.id()))n.position({x:80+column*440,y:y+height/2});y+=height});
 }
}

export function attachLabelAvoidance(cy:Core,host:HTMLElement){
 let timer:ReturnType<typeof setTimeout>|undefined,dead=false,runs=0;
 const apply=()=>{
  if(dead||cy.destroyed()||!cy.nodes().length)return;
  const start=performance.now(),zoom=cy.zoom();
  cy.batch(()=>{cy.nodes().removeClass('label-hidden');cy.nodes().style({'font-size':Math.max(12,12/zoom),'text-max-width':`${120/zoom}px`,'text-margin-y':8/zoom,'text-margin-x':0,'text-valign':'bottom','text-halign':'center','text-background-padding':`${3/zoom}px`})});
  const nodes=cy.nodes().toArray().sort((a,b)=>{
   const priority=(n:NodeSingular)=>n.selected()?100:n.hasClass('search-hit')?90:n.hasClass('hot')?80:n.hasClass('neighbor-focus')?60:n.degree()>0?30:10;
   return priority(b)-priority(a)||a.id().localeCompare(b.id());
  });
  const bodies=nodes.map(n=>({id:n.id(),box:n.renderedBoundingBox({includeLabels:false,includeOverlays:false,includeUnderlays:false})}));
  const accepted:Box[]=[];let hidden=0;
  const suppressed:NodeSingular[]=[];
  for(const n of nodes){
   let placed=false;
   // A selected/hovered label must get first choice of free space, rather than
   // disappearing merely because its default bottom position covers a node.
   const placements=[{v:'bottom',h:'center',x:0,y:8},{v:'center',h:'left',x:-8,y:0},{v:'center',h:'right',x:8,y:0},{v:'top',h:'center',x:0,y:-8}] as const;
   for(const p of placements){
    n.style({'text-valign':p.v,'text-halign':p.h,'text-margin-x':p.x/zoom,'text-margin-y':p.y/zoom});
    const box=n.renderedBoundingBox({includeNodes:false,includeEdges:false,includeLabels:true,includeOverlays:false,includeUnderlays:false});
    if(!Number.isFinite(box.x1)||box.x2<=box.x1)continue;
    if(box.x1<0||box.y1<0||box.x2>cy.width()||box.y2>cy.height())continue;
    if(bodies.some(b=>b.id!==n.id()&&overlap(box,b.box,2))||accepted.some(b=>overlap(box,b,4)))continue;
    accepted.push(box);placed=true;break;
   }
   if(!placed){suppressed.push(n);hidden++}
  }
  cy.batch(()=>suppressed.forEach(n=>n.addClass('label-hidden')));
  host.dataset.labelsHidden=String(hidden);host.dataset.labelsVisible=String(accepted.length);host.dataset.collisionMs=(performance.now()-start).toFixed(3);host.dataset.collisionRuns=String(++runs);
 };
 const schedule=()=>{clearTimeout(timer);timer=setTimeout(apply,85)};
 cy.on('layoutstop zoom pan dragfree select unselect mouseover mouseout',schedule);
 return {schedule,apply,destroy(){dead=true;clearTimeout(timer);cy.off('layoutstop zoom pan dragfree select unselect mouseover mouseout',schedule)}};
}
