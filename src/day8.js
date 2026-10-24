
// ---------- 第 8 天：用 ReLU 拼曲线 ----------
const UA_F=[x=>Math.sin(2*x)+0.3*x,x=>Math.abs(x)-1,x=>x<-1?-1:x<1?0.3:1.2];
function uaFit(h,f){const xs=Array.from({length:121},(_,i)=>-3+i*0.05),ks=Array.from({length:h},(_,i)=>-3+6*(i+0.5)/h);
  const feat=x=>[1,x,...ks.map(k=>Math.max(0,x-k))];const n=h+2,A=Array.from({length:n},()=>new Float64Array(n+1));
  for(const x of xs){const p=feat(x),y=f(x);for(let i=0;i<n;i++){for(let j=0;j<n;j++)A[i][j]+=p[i]*p[j];A[i][n]+=p[i]*y}}for(let i=0;i<n;i++)A[i][i]+=1e-6;
  for(let i=0;i<n;i++){let m=i;for(let r=i+1;r<n;r++)if(Math.abs(A[r][i])>Math.abs(A[m][i]))m=r;[A[i],A[m]]=[A[m],A[i]];for(let r=0;r<n;r++){if(r===i)continue;const c=A[r][i]/A[i][i];for(let j=i;j<=n;j++)A[r][j]-=c*A[i][j]}}
  const w=A.map((r,i)=>r[n]/r[i]);const pred=x=>feat(x).reduce((s,v,i)=>s+v*w[i],0);let mse=0;for(const x of xs)mse+=(pred(x)-f(x))**2;return{ks,w,pred,mse:mse/xs.length}}
function uaRender(){const host=$('#uaHost');if(!host.offsetParent)return;const C=colors(),h=+$('#uaH').value,f=UA_F[+$('#uaF').value];$('#uaHv').textContent=h;const r=uaFit(h,f);
  const W=Math.max(260,host.clientWidth),H=Math.round(Math.min(300,W*.7)),ctx=fitCanvas($('#uaCv'),W,H),sx=x=>(x+3.2)/6.4*W,sy=y=>H/2-y*H/5.2;
  ctx.strokeStyle=C.line;ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(0,sy(0));ctx.lineTo(W,sy(0));ctx.moveTo(sx(0),0);ctx.lineTo(sx(0),H);ctx.stroke();
  ctx.save();ctx.beginPath();ctx.rect(0,0,W,H);ctx.clip();ctx.globalAlpha=.35;ctx.strokeStyle=C.muted;ctx.lineWidth=1;
  r.ks.forEach((k,i)=>{const wk=r.w[i+2];ctx.beginPath();for(let q=0;q<=100;q++){const x=-3.2+6.4*q/100,y=wk*Math.max(0,x-k);q?ctx.lineTo(sx(x),sy(y)):ctx.moveTo(sx(x),sy(y))}ctx.stroke()});ctx.globalAlpha=1;
  const line=(g,col,w,dash)=>{ctx.strokeStyle=col;ctx.lineWidth=w;ctx.setLineDash(dash);ctx.beginPath();for(let q=0;q<=320;q++){const x=-3+6*q/320;q?ctx.lineTo(sx(x),sy(g(x))):ctx.moveTo(sx(x),sy(g(x)))}ctx.stroke();ctx.setLineDash([])};
  line(f,C.accent,3,[]);line(r.pred,C.pos,2,[6,4]);r.ks.forEach(k=>{ctx.fillStyle=C.pos;ctx.beginPath();ctx.arc(sx(k),sy(r.pred(k)),2.5,0,7);ctx.fill()});ctx.restore();
  ctx.font=bodyFont(12);ctx.fillStyle=C.ink;ctx.fillText('— 目标',8,16);ctx.fillText('- - MLP 输出',8,32);
  $('#uaCalc').textContent=`隐藏层：${h} 个 ReLU 单元，第 i 个算 ReLU(x − kᵢ)\n输出层：o = b + a·x + Σ wᵢ·ReLU(x − kᵢ)\n参数：${h} 个折点 + ${h+2} 个输出层权重\n\n均方误差 = ${r.mse.toExponential(2)}\n${r.mse<1e-3?'已经贴得很紧了。':h<4?'单元太少，只能拼出几段直线。':'再多加几个单元会更贴近。'}\n\n这里折点 kᵢ 是固定排开的，只求输出层权重。\n真正训练 MLP 时，折点（来自第一层的权重和偏置）也会一起学。`}
$('#uaH').addEventListener('input',uaRender);$('#uaF').onchange=uaRender;
// ---------- 第 8 天：三个激活函数 ----------
function actRender(){const x=+$('#actX').value;$('#actXv').textContent=x.toFixed(1);const s=1/(1+Math.exp(-x)),t=Math.tanh(x);
  $('#actCalc').textContent=`函数        值          导数\nReLU     ${Math.max(0,x).toFixed(4).padStart(8)}     ${(x>0?1:0).toFixed(4).padStart(8)}   ${x>0?'导数是 1，梯度原样通过':'导数是 0，梯度被挡住'}\nsigmoid  ${s.toFixed(4).padStart(8)}     ${(s*(1-s)).toFixed(4).padStart(8)}   ${Math.abs(x)>4?'导数接近 0：饱和区，梯度几乎消失':'最大 0.25，在 x = 0 处'}\ntanh     ${t.toFixed(4).padStart(8)}     ${(1-t*t).toFixed(4).padStart(8)}   ${Math.abs(x)>3?'导数接近 0：饱和区':'最大 1，在 x = 0 处'}\n\n叠 10 层 sigmoid、每层导数都取最大值 0.25：\n0.25^10 = ${(0.25**10).toExponential(2)}，梯度到第一层时几乎没了。`}
$('#actX').addEventListener('input',actRender);
