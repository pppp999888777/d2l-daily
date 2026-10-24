// ---------- 第 15 天：Nadaraya-Watson 的查询和核宽度 ----------
const NW=(()=>{const g=sgRng(21);const u=(()=>{let s=99;return()=>{s=s+0x6D2B79F5>>>0;let t=Math.imul(s^s>>>15,1|s);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}})();const f=x=>2*Math.sin(x)+x**0.8;const x=Array.from({length:50},()=>u()*5).sort((a,b)=>a-b);return{f,x,y:x.map(v=>f(v)+0.5*g())}})();
function nwPred(q,w){const s=NW.x.map(xi=>-(((q-xi)*w)**2)/2),m=Math.max(...s),e=s.map(v=>Math.exp(v-m)),z=e.reduce((a,b)=>a+b,0),a=e.map(v=>v/z);return{a,y:a.reduce((t,v,i)=>t+v*NW.y[i],0)}}
function nwRender(){const host=$('#nwHost');if(!host.offsetParent)return;const C=colors(),q=+$('#nwQ').value,w=+$('#nwW').value;$('#nwQv').textContent=q.toFixed(2);$('#nwWv').textContent=w.toFixed(2);
  const W=Math.max(260,host.clientWidth),H=Math.round(Math.min(320,W*.75)),ctx=fitCanvas($('#nwCv'),W,H),pl=8,pr=8,top=10,bh=60,sx=x=>pl+x/5*(W-pl-pr),sy=y=>top+(5-y)/6*(H-top-bh-20);
  ctx.strokeStyle=C.line;ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(pl,sy(0));ctx.lineTo(W-pr,sy(0));ctx.stroke();
  ctx.strokeStyle=C.muted;ctx.lineWidth=2;ctx.beginPath();for(let k=0;k<=200;k++){const x=5*k/200;k?ctx.lineTo(sx(x),sy(NW.f(x))):ctx.moveTo(sx(x),sy(NW.f(x)))}ctx.stroke();
  ctx.strokeStyle=C.pos;ctx.lineWidth=2.5;ctx.beginPath();for(let k=0;k<=200;k++){const x=5*k/200,y=nwPred(x,w).y;k?ctx.lineTo(sx(x),sy(y)):ctx.moveTo(sx(x),sy(y))}ctx.stroke();
  const r=nwPred(q,w),mx=Math.max(...r.a);NW.x.forEach((x,i)=>{ctx.fillStyle=C.accent;ctx.globalAlpha=.25+.75*r.a[i]/mx;ctx.beginPath();ctx.arc(sx(x),sy(NW.y[i]),3,0,7);ctx.fill();ctx.globalAlpha=1;const h=r.a[i]/mx*(bh-8);ctx.fillStyle=C.accent;ctx.fillRect(sx(x)-2,H-6-h,4,h)});
  ctx.strokeStyle=C.ink;ctx.setLineDash([4,4]);ctx.beginPath();ctx.moveTo(sx(q),0);ctx.lineTo(sx(q),H);ctx.stroke();ctx.setLineDash([]);ctx.fillStyle=C.pos;ctx.beginPath();ctx.arc(sx(q),sy(r.y),5,0,7);ctx.fill();
  ctx.font=bodyFont(11);ctx.fillStyle=C.muted;ctx.fillText('注意力权重',pl+2,H-bh+4);
  const top3=r.a.map((a,i)=>[a,i]).sort((a,b)=>b[0]-a[0]).slice(0,3),ent=-r.a.reduce((s,a)=>s+(a>0?a*Math.log(a):0),0);
  $('#nwCalc').textContent=`查询 x = ${q.toFixed(2)}，w = ${w.toFixed(2)}\n预测 ${r.y.toFixed(3)}，真实值 ${NW.f(q).toFixed(3)}\n\n权重最大的 3 个训练点：\n${top3.map(([a,i])=>`  x = ${NW.x[i].toFixed(2)}  y = ${NW.y[i].toFixed(2)}  权重 ${a.toFixed(3)}`).join('\n')}\n相当于在约 ${Math.exp(ent).toFixed(1)} 个点之间平均（50 个点全平均时是 50）\n\n${w<0.3?'w 很小：几乎所有点权重一样，接近平均汇聚，预测是一条平线。':w>6?'w 很大：只看最近的一两个点，预测曲线跟着噪声抖。':'w 适中：主要看附近的点，曲线比较平滑地跟着真实函数。'}`}
['#nwQ','#nwW'].forEach(s=>$(s).addEventListener('input',nwRender));
// ---------- 第 15 天：缩放点积 ----------
const SD={seed:1};
function sdRender(){const host=$('#sdHost');if(!host.offsetParent)return;const d=+$('#sdD').value;$('#sdDv').textContent=d;const g=sgRng(SD.seed*7919+d);
  const q=Array.from({length:d},g),ks=Array.from({length:8},()=>Array.from({length:d},g)),sc=ks.map(k=>k.reduce((s,v,i)=>s+v*q[i],0)),sm=a=>{const m=Math.max(...a),e=a.map(v=>Math.exp(v-m)),z=e.reduce((x,y)=>x+y,0);return e.map(v=>v/z)};
  const p0=sm(sc),p1=sm(sc.map(v=>v/Math.sqrt(d))),C=colors();
  const W=Math.max(260,host.clientWidth),H=200,ctx=fitCanvas($('#sdCv'),W,H),pl=10,pr=10,bw=(W-pl-pr)/8;ctx.font=bodyFont(11);
  for(let i=0;i<8;i++){const x=pl+i*bw;[[p0[i],C.neg,0],[p1[i],C.accent,1]].forEach(([p,col,j])=>{const h=p*(H-46);ctx.fillStyle=col;ctx.fillRect(x+bw*(0.12+0.38*j),H-24-h,bw*0.34,h)});ctx.fillStyle=C.muted;ctx.textAlign='center';ctx.fillText('键 '+(i+1),x+bw/2,H-8)}
  ctx.textAlign='left';ctx.fillStyle=C.neg;ctx.fillRect(pl,6,12,10);ctx.fillStyle=C.ink;ctx.fillText('不缩放 qᵀk',pl+16,15);ctx.fillStyle=C.accent;ctx.fillRect(pl+110,6,12,10);ctx.fillStyle=C.ink;ctx.fillText('除以 √d',pl+126,15);
  const sdv=a=>Math.sqrt(a.reduce((s,v)=>s+v*v,0)/a.length);
  $('#sdCalc').textContent=`d = ${d}\n8 个分数的标准差：不缩放 ${sdv(sc).toFixed(2)}，除以 √d 后 ${sdv(sc.map(v=>v/Math.sqrt(d))).toFixed(2)}\n最大的权重：不缩放 ${Math.max(...p0).toFixed(3)}，除以 √d 后 ${Math.max(...p1).toFixed(3)}\n\n${d<4?'d 很小时两者差别不大。':Math.max(...p0)>0.95?'不缩放时几乎所有权重都压在一个键上，softmax 饱和了：其他键的梯度接近 0，学不动。':'d 越大，不缩放的分数越分散，权重越尖。'}`}
$('#sdD').addEventListener('input',sdRender);$('#sdGo').onclick=()=>{SD.seed++;sdRender()};
