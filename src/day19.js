// ---------- 第 19 天：OPE 估计器游乐场 ----------
const OE={seed:5};
function oeSim(){const T=10**+$('#oeT').value,n=+$('#oeN').value,clip=+$('#oeC').value,qe=+$('#oeQ').value;$('#oeTv').textContent=T.toFixed(3);$('#oeNv').textContent=n;$('#oeQv').textContent=qe.toFixed(2);
  const g=sgRng(OE.seed),u=(()=>{let s=OE.seed*977;return()=>{s=s+0x6D2B79F5>>>0;let t=Math.imul(s^s>>>15,1|s);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}})(),K=5;
  const rm=p=>0.3*Math.exp(-(((p-0.6)/0.2)**2)),Q=p=>(1-qe)*rm(p)+qe*(0.6*rm(p)+0.05),sm=a=>{const m=Math.max(...a),e=a.map(v=>Math.exp(v-m)),z=e.reduce((x,y)=>x+y,0);return e.map(v=>v/z)};
  const mu=p=>sm(p.map(v=>8*v)),pi=p=>sm(p.map(v=>-((v-0.7)**2)/T),0);
  let V=0;for(let i=0;i<20000;i++){const p=Array.from({length:K},()=>u()*0.9+0.05),pp=pi(p);V+=pp.reduce((s,v,k)=>s+v*rm(p[k]),0)}V/=20000;
  const R={IPS:[],SNIPS:[],DM:[],DR:[]},ess=[];
  for(let rep=0;rep<200;rep++){let sw=0,sw2=0,swr=0,sdm=0,sdr=0;for(let i=0;i<n;i++){const p=Array.from({length:K},()=>u()*0.9+0.05),m=mu(p),pp=pi(p);let c=u(),a=0;while(a<K-1&&(c-=m[a])>=0)a++;
      const r=rm(p[a])+0.05*g();let w=pp[a]/m[a];if(clip)w=Math.min(w,clip);const dm=pp.reduce((s,v,k)=>s+v*Q(p[k]),0);sw+=w;sw2+=w*w;swr+=w*r;sdm+=dm;sdr+=dm+w*(r-Q(p[a]))}
    R.IPS.push(swr/n);R.SNIPS.push(swr/sw);R.DM.push(sdm/n);R.DR.push(sdr/n);ess.push(sw*sw/sw2/n)}
  OE.res={V,R,ess:ess.reduce((a,b)=>a+b,0)/ess.length,T,clip,qe,n};oeDraw()}
function oeDraw(){const host=$('#oeHost');if(!host.offsetParent||!OE.res)return;const{V,R,ess,T,clip,qe,n}=OE.res,C=colors(),names=['IPS','SNIPS','DM','DR'],cols=[C.neg,C.accent,C.muted,C.pos];
  const all=names.flatMap(k=>R[k]),lo=Math.min(V,...all),hi=Math.max(V,...all),pad=(hi-lo)*0.08||0.01,y0=lo-pad,y1=hi+pad;
  const W=Math.max(260,host.clientWidth),H=260,ctx=fitCanvas($('#oeCv'),W,H),pl=48,pb=24,cw=(W-pl-8)/4,sy=y=>8+(y1-y)/(y1-y0)*(H-8-pb);
  ctx.font=MONO;ctx.fillStyle=C.muted;ctx.textAlign='right';ctx.textBaseline='middle';ctx.strokeStyle=C.line;for(let k=0;k<=4;k++){const y=y0+(y1-y0)*k/4;ctx.beginPath();ctx.moveTo(pl,sy(y));ctx.lineTo(W-8,sy(y));ctx.stroke();ctx.fillText(y.toFixed(3),pl-4,sy(y))}
  ctx.strokeStyle=C.ink;ctx.setLineDash([5,4]);ctx.beginPath();ctx.moveTo(pl,sy(V));ctx.lineTo(W-8,sy(V));ctx.stroke();ctx.setLineDash([]);
  names.forEach((k,i)=>{ctx.fillStyle=cols[i];ctx.globalAlpha=.55;R[k].forEach((v,j)=>{ctx.beginPath();ctx.arc(pl+cw*(i+0.5)+((j*37)%41-20)*cw/60,sy(v),2.2,0,7);ctx.fill()});ctx.globalAlpha=1;ctx.fillStyle=C.ink;ctx.textAlign='center';ctx.textBaseline='alphabetic';ctx.font=bodyFont(12);ctx.fillText(k,pl+cw*(i+0.5),H-6);ctx.font=MONO});
  const st=a=>{const m=a.reduce((x,y)=>x+y,0)/a.length,s=Math.sqrt(a.reduce((x,y)=>x+(y-m)**2,0)/(a.length-1));return[m-V,s]};
  $('#oeCalc').textContent=`真实 V(π) = ${V.toFixed(4)}（虚线）；n = ${n}，温度 ${T.toFixed(3)}，${clip?'截断到 '+clip:'不截断'}，模型误差 ${qe.toFixed(2)}\n平均 ESS = ${ess.toFixed(3)}\n\n         偏差      标准差\n${names.map(k=>{const[b,s]=st(R[k]);return`${k.padEnd(6)} ${(b>=0?'+':'')+b.toFixed(4)}   ${s.toFixed(4)}`}).join('\n')}\n\n${ess<0.15?'ESS 很低：新策略和旧策略差得太远，估计主要靠少数几条记录。':''}${qe===0?'模型完全正确：DM 和 DR 都没有偏差，DR 方差最小。':qe>0.7?'模型误差很大：DM 偏差明显，DR 仍然基本无偏。':''}${clip&&clip<=5?'\n截断很狠：IPS 方差变小，但出现了偏差。':''}`}
$('#oeGo').onclick=()=>{OE.seed++;oeSim()};['#oeT','#oeN','#oeQ'].forEach(s=>$(s).addEventListener('change',oeSim));['#oeT','#oeN','#oeQ'].forEach(s=>$(s).addEventListener('input',()=>{$('#oeTv').textContent=(10**+$('#oeT').value).toFixed(3);$('#oeNv').textContent=$('#oeN').value;$('#oeQv').textContent=(+$('#oeQ').value).toFixed(2)}));$('#oeC').onchange=oeSim;
function oeRender(){if(!OE.res)oeSim();else oeDraw()}
