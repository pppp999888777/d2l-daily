// ---------- 第 10 天：信号穿过 L 层 ----------
function sgRng(seed){let s=seed>>>0;const u=()=>{s=s+0x6D2B79F5>>>0;let t=Math.imul(s^s>>>15,1|s);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296};return()=>{let a=0;while(!a)a=u();return Math.sqrt(-2*Math.log(a))*Math.cos(2*Math.PI*u())}}
const SG_ACT={sigmoid:[x=>1/(1+Math.exp(-x)),(x,y)=>y*(1-y)],tanh:[Math.tanh,(x,y)=>1-y*y],relu:[x=>x>0?x:0,x=>x>0?1:0]};
function sgSim(act,init,L){const n=64,B=64,g=sgRng(7),[f,df]=SG_ACT[act],sd={small:0.01,one:1,xavier:Math.sqrt(2/(n+n)),he:Math.sqrt(2/n)}[init];
  const Ws=[],Zs=[],Hs=[];let h=Array.from({length:B*n},()=>g());Hs.push(h);
  for(let l=0;l<L;l++){const W=Float64Array.from({length:n*n},()=>g()*sd),z=new Float64Array(B*n);for(let b=0;b<B;b++)for(let i=0;i<n;i++){const hv=h[b*n+i];if(!hv)continue;for(let j=0;j<n;j++)z[b*n+j]+=hv*W[i*n+j]}
    const o=Float64Array.from(z,v=>f(v));Ws.push(W);Zs.push(z);Hs.push(o);h=o}
  const fstd=Hs.slice(1).map(a=>{let m=0;for(const v of a)m+=v;m/=a.length;let s=0;for(const v of a)s+=(v-m)**2;return Math.sqrt(s/a.length)});
  let gh=Array.from({length:B*n},()=>g()/B);const gn=new Array(L);
  for(let l=L-1;l>=0;l--){const z=Zs[l],o=Hs[l+1],hin=Hs[l],W=Ws[l],gz=Float64Array.from(gh,(v,k)=>v*df(z[k],o[k]));let s=0;
    for(let i=0;i<n;i++)for(let j=0;j<n;j++){let a=0;for(let b=0;b<B;b++)a+=hin[b*n+i]*gz[b*n+j];s+=a*a}gn[l]=Math.sqrt(s);
    const nh=new Float64Array(B*n);for(let b=0;b<B;b++)for(let i=0;i<n;i++){let a=0;for(let j=0;j<n;j++)a+=gz[b*n+j]*W[i*n+j];nh[b*n+i]=a}gh=nh}
  return{fstd,gn,sd}}
function sgRender(){const host=$('#sgHost');if(!host.offsetParent)return;const act=$('#sgAct').value,init=$('#sgInit').value,L=+$('#sgL').value;$('#sgLv').textContent=L;
  const r=sgSim(act,init,L),xs=Array.from({length:L},(_,i)=>i+1),cl=v=>Math.max(1e-30,Math.min(1e30,v));
  host.innerHTML='<canvas id="sgCv" aria-label="每一层的前向信号和梯度大小"></canvas>';
  drawPlot($('#sgCv'),{series:[{x:xs,y:r.fstd.map(cl),label:'前向：输出的标准差'},{x:xs,y:r.gn.map(cl),label:'反向：权重梯度的大小'}],hlines:[],xlabel:'第几层',ylabel:'',legend:true,yscale:'log'});
  const ratio=(a,b)=>a/b,fr=ratio(r.fstd[L-1],r.fstd[0]),gr=ratio(r.gn[0],r.gn[L-1]),e=v=>v.toExponential(1);
  const verdict=v=>v<1e-3?'消失了':v>1e3?'爆炸了':v<0.1||v>10?'有明显变化':'基本保持住了';
  $('#sgCalc').textContent=`每层权重的标准差 σ = ${r.sd.toPrecision(3)}${init==='xavier'?'（Xavier：√(2/(64+64))）':init==='he'?'（He：√(2/64)）':''}\n\n前向信号：第 1 层 ${e(r.fstd[0])}，第 ${L} 层 ${e(r.fstd[L-1])}，${verdict(fr)}\n反向梯度：最后一层 ${e(r.gn[L-1])}，第 1 层 ${e(r.gn[0])}，${verdict(gr)}\n\n${act==='sigmoid'?'sigmoid 的导数最大 0.25，层数一多梯度就消失；而且它的输出均值不是 0，信号也很难保持。':act==='relu'&&init==='he'?'ReLU 会把一半的输出变成 0，方差减半；He 初始化把 σ² 乘 2 正好补回来。':act==='relu'&&init==='xavier'?'Xavier 没考虑 ReLU 砍掉一半，所以每层信号缩小一点，层数多了就看得出来。试试 He。':init==='one'?'σ = 1 太大：每层把信号放大约 √64 = 8 倍（ReLU 是约 5.7 倍）。':init==='small'?'σ = 0.01 太小：每层把信号缩小到约 0.01 × √64 = 0.08 倍。':'tanh 在 0 附近导数是 1，配 Xavier 比较稳。'}`}
['#sgAct','#sgInit'].forEach(s=>$(s).onchange=sgRender);$('#sgL').addEventListener('input',sgRender);
// ---------- 第 10 天：协变量偏移和重要性加权 ----------
const SH={n:300,s:0.6,mS:-0.5};
(function(){const g=sgRng(11);SH.x=Array.from({length:SH.n},()=>SH.mS+SH.s*g());SH.y=SH.x.map(x=>x*x/2+0.1*g())})();
function shFit(w){let sw=0,sx=0,sy=0,sxx=0,sxy=0;SH.x.forEach((x,i)=>{const v=w[i];sw+=v;sx+=v*x;sy+=v*SH.y[i];sxx+=v*x*x;sxy+=v*x*SH.y[i]});const mx=sx/sw,my=sy/sw,b1=(sxy/sw-mx*my)/(sxx/sw-mx*mx);return[my-b1*mx,b1]}
function shRender(){const host=$('#shHost');if(!host.offsetParent)return;const C=colors(),mT=+$('#shM').value,clip=+$('#shClip').value;$('#shMv').textContent=mT.toFixed(2);
  const pdf=(x,m)=>Math.exp(-((x-m)**2)/(2*SH.s*SH.s)),beta=SH.x.map(x=>Math.min(clip,pdf(x,mT)/pdf(x,SH.mS)));
  const fp=shFit(SH.x.map(()=>1)),fw=shFit(beta);
  // 目标分布上的期望 MSE：在目标分布上数值积分（噪声方差 0.01 加上去）
  let mp=0,mw=0,z=0;for(let k=0;k<=400;k++){const x=mT-4*SH.s+8*SH.s*k/400,p=pdf(x,mT),t=x*x/2;z+=p;mp+=p*(fp[0]+fp[1]*x-t)**2;mw+=p*(fw[0]+fw[1]*x-t)**2}mp=mp/z+0.01;mw=mw/z+0.01;
  const sw=beta.reduce((a,b)=>a+b,0),sw2=beta.reduce((a,b)=>a+b*b,0),ess=sw*sw/sw2,nclip=beta.filter(b=>b>=clip).length;
  const W=Math.max(260,host.clientWidth),H=Math.round(Math.min(320,W*.72)),ctx=fitCanvas($('#shCv'),W,H),x0=-2.4,x1=3.4,y0=-0.8,y1=3.2,sx=x=>(x-x0)/(x1-x0)*W,sy=y=>H-26-(y-y0)/(y1-y0)*(H-36);
  ctx.strokeStyle=C.line;ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(0,sy(0));ctx.lineTo(W,sy(0));ctx.moveTo(sx(0),0);ctx.lineTo(sx(0),H);ctx.stroke();
  const dens=(m,col)=>{ctx.fillStyle=col;ctx.globalAlpha=.22;ctx.beginPath();ctx.moveTo(sx(x0),H);for(let k=0;k<=200;k++){const x=x0+(x1-x0)*k/200;ctx.lineTo(sx(x),H-pdf(x,m)*40)}ctx.lineTo(sx(x1),H);ctx.fill();ctx.globalAlpha=1};
  dens(SH.mS,C.accent);dens(mT,C.pos);
  ctx.save();ctx.beginPath();ctx.rect(0,0,W,H);ctx.clip();
  ctx.strokeStyle=C.muted;ctx.lineWidth=2;ctx.beginPath();for(let k=0;k<=200;k++){const x=x0+(x1-x0)*k/200;k?ctx.lineTo(sx(x),sy(x*x/2)):ctx.moveTo(sx(x),sy(x*x/2))}ctx.stroke();
  SH.x.forEach((x,i)=>{ctx.fillStyle=C.accent;ctx.globalAlpha=Math.min(1,.15+.5*beta[i]/Math.max(...beta));ctx.beginPath();ctx.arc(sx(x),sy(SH.y[i]),2.2,0,7);ctx.fill()});ctx.globalAlpha=1;
  const ln=(c,col,dash)=>{ctx.strokeStyle=col;ctx.lineWidth=2;ctx.setLineDash(dash);ctx.beginPath();ctx.moveTo(sx(x0),sy(c[0]+c[1]*x0));ctx.lineTo(sx(x1),sy(c[0]+c[1]*x1));ctx.stroke();ctx.setLineDash([])};
  ln(fp,C.ink,[6,4]);ln(fw,C.pos,[]);ctx.restore();
  ctx.font=bodyFont(12);[[C.ink,[6,4],'不加权的直线'],[C.pos,[],'加权的直线']].forEach(([c,d,t],i)=>{ctx.strokeStyle=c;ctx.setLineDash(d);ctx.beginPath();ctx.moveTo(8,14+i*16);ctx.lineTo(28,14+i*16);ctx.stroke();ctx.setLineDash([]);ctx.fillStyle=C.ink;ctx.fillText(t,34,18+i*16)});ctx.fillStyle=C.muted;ctx.fillText('蓝点和蓝色区域：源分布　橙色区域：目标分布',8,50);
  $('#shCalc').textContent=`源分布中心 −0.50，目标分布中心 ${mT.toFixed(2)}，两边标准差都是 0.6\nβ(x) = p_T(x) / p_S(x)，这里两个分布已知，直接算${clip<1e8?`，再截断到 ${clip}（被截断的样本 ${nclip} 个）`:''}\n\n                目标分布上的 MSE\n不加权          ${mp.toFixed(4)}\n加权            ${mw.toFixed(4)}\n\n有效样本量 ESS = (Σβ)² / Σβ² = ${ess.toFixed(1)}（一共 ${SH.n} 个样本）\n${ess<20?'ESS 很小：只有极少数样本的权重大，加权结果很不稳定。两个分布离得太远时，重要性加权也救不了。':ess<SH.n*0.5?'权重比较不均匀，加权估计的方差会变大。':'两个分布很接近，权重都差不多，加不加权区别不大。'}`}
$('#shM').addEventListener('input',shRender);$('#shClip').onchange=shRender;
