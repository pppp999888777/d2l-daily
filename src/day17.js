// ---------- 第 17 天：层规范化和批量规范化 ----------
const LN={X:[[1,2,3],[2,4,9],[0,1,1],[5,5,8]]};
function lnNorm(X,mode){const R=X.length,C=X[0].length,Y=X.map(r=>r.slice());if(mode==='ln'){X.forEach((r,i)=>{const m=r.reduce((a,b)=>a+b,0)/C,v=r.reduce((a,b)=>a+(b-m)**2,0)/C;r.forEach((x,j)=>Y[i][j]=(x-m)/Math.sqrt(v+1e-5))})}
  else{for(let j=0;j<C;j++){const col=X.map(r=>r[j]),m=col.reduce((a,b)=>a+b,0)/R,v=col.reduce((a,b)=>a+(b-m)**2,0)/R;X.forEach((r,i)=>Y[i][j]=(r[j]-m)/Math.sqrt(v+1e-5))}}return Y}
function lnRender(){const mode=$('#lnMode').value,Y=lnNorm(LN.X,mode),grp=(i,j)=>mode==='ln'?i:j,cols=['--accent','--pos','--neg','--muted'];
  const cell=(v,i,j,inp)=>`<button class="lnc" ${inp?`data-i="${i}" data-j="${j}"`:'disabled'} style="background:color-mix(in srgb,var(${cols[grp(i,j)%4]}) 18%,var(--surface))">${inp?v:v.toFixed(2)}</button>`;
  $('#lnIn').innerHTML='<div class="small muted">输入（点格子改数）</div>'+LN.X.map((r,i)=>`<div class="lnrow">${r.map((v,j)=>cell(v,i,j,true)).join('')}</div>`).join('');
  $('#lnOut').innerHTML='<div class="small muted">输出</div>'+Y.map((r,i)=>`<div class="lnrow">${r.map((v,j)=>cell(v,i,j,false)).join('')}</div>`).join('');
  $$('#lnIn .lnc').forEach(b=>b.onclick=()=>{const i=+b.dataset.i,j=+b.dataset.j;LN.X[i][j]=(LN.X[i][j]+3)%10;lnRender()});
  const rowM=Y.map(r=>r.reduce((a,b)=>a+b,0)/r.length),colM=Y[0].map((_,j)=>Y.reduce((a,r)=>a+r[j],0)/Y.length);
  $('#lnCalc').textContent=`同一种颜色的格子是一起标准化的一组。\n输出每一行的均值：${rowM.map(v=>v.toFixed(2)).join('，')}\n输出每一列的均值：${colM.map(v=>v.toFixed(2)).join('，')}\n\n${mode==='ln'?'层规范化：每个样本只用自己的数，和批量里其他样本无关，批量大小是 1 也能用。\n这也是 Transformer 和 SAKT 用它的原因：序列里每个位置单独规范化。':'批量规范化：每个特征要用到整个批量的统计量，批量太小时均值和方差很不稳定；\n评估时还要换成训练时累积的平均值（running_mean / running_var）。'}`}
$('#lnMode').onchange=lnRender;$('#lnRand').onclick=()=>{LN.X=LN.X.map(r=>r.map(()=>Math.floor(Math.random()*10)));lnRender()};
// ---------- 第 17 天：AKT 的距离衰减 ----------
function dcRender(){const host=$('#dcHost');if(!host.offsetParent)return;const th=+$('#dcT').value;$('#dcTv').textContent=th.toFixed(2);const n=20,dist=Array.from({length:n},(_,j)=>n-1-j),s=dist.map(d=>-th*d),m=Math.max(...s),e=s.map(v=>Math.exp(v-m)),z=e.reduce((a,b)=>a+b,0),w=e.map(v=>v/z);
  const C=colors(),W=Math.max(260,host.clientWidth),H=180,ctx=fitCanvas($('#dcCv'),W,H),pl=8,bw=(W-2*pl)/n;ctx.font=bodyFont(11);ctx.textAlign='center';const mx=Math.max(...w);
  w.forEach((v,j)=>{const h=v/mx*(H-40);ctx.fillStyle=C.accent;ctx.fillRect(pl+j*bw+bw*0.15,H-20-h,bw*0.7,h);if(j%3===1||j===n-1){ctx.fillStyle=C.muted;ctx.fillText(String(dist[j]),pl+j*bw+bw/2,H-6)}});ctx.textAlign='left';ctx.fillStyle=C.muted;ctx.fillText('柱子下面的数：离当前多少步',pl,12);
  const near=w.slice(-5).reduce((a,b)=>a+b,0);
  $('#dcCalc').textContent=`θ = ${th.toFixed(2)}，权重 ∝ exp(−θ × 距离)\n最近一题的权重 ${w[n-1].toFixed(3)}，最远一题 ${w[0].toFixed(3)}\n最近 5 题加起来占 ${(near*100).toFixed(0)}%（没有衰减时是 25%）\n\n${th<0.01?'θ ≈ 0：没有衰减，20 道题平分注意力，等于 SAKT。':th>0.4?'θ 很大：几乎只看最近一两题。':'θ 适中：越久以前的题越不重要，模拟“遗忘”。'}\nθ 是训练出来的：数据里遗忘越明显，学到的 θ 越大。`}
$('#dcT').addEventListener('input',dcRender);
