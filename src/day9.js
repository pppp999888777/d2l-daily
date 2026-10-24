// ---------- 第 9 天：论文的数据划分 ----------
const SP_DS=[['ASSISTments 2009',3841,{train:3718,val:411,test:1050}],['EdNet',5000,{train:3600,val:400,test:1000}],['Algebra 2005',574,{train:1604,val:183,test:448}],['Junyi',5000,{train:5159,val:571,test:1414}]];
function spRender(){const[nm,n,w]=SP_DS[+$('#spDs').value],nt=Math.max(1,Math.round(0.2*n)),rest=n-nt,nv=Math.max(1,Math.round(0.1*rest)),ntr=rest-nv;
  $('#spBar').innerHTML=[['tr','训练',ntr],['va','验证',nv],['te','测试',nt]].map(([c,l,k])=>`<div class="${c}" style="flex:${k}"><b>${l}</b> ${k}</div>`).join('');
  $('#spCalc').textContent=`${nm}：一共 ${n} 个学生\n先随机打乱，取 20% = ${nt} 个做测试集\n剩下 ${rest} 个里取 10% = ${nv} 个做验证集\n其余 ${ntr} 个做训练集\n\n每个学生的做题序列再切成 100 步一段（窗口）：\n训练 ${w.train} 段，验证 ${w.val} 段，测试 ${w.test} 段\n\n代码：experiment.py 第 648 到 650 行（prep_world）\n训练集：拿来更新参数\n验证集：每个 epoch 算一次 BCE，挑最好的检查点\n测试集：最后只看一次，报告在论文表格里`}
$('#spDs').innerHTML=SP_DS.map((d,i)=>`<option value="${i}">${d[0]}（${d[1]} 个学生）</option>`).join('');$('#spDs').onchange=spRender;
// ---------- 第 9 天：λ 把最优解拉向原点 ----------
const RD={A:[[3,1.2],[1.2,0.8]],ws:[2,1.4]};
function rdSolve(l){const[[a,b],[c,d]]=RD.A,p=a+l,s=d+l,det=p*s-b*c,r0=a*RD.ws[0]+b*RD.ws[1],r1=c*RD.ws[0]+d*RD.ws[1];return[(s*r0-b*r1)/det,(p*r1-c*r0)/det]}
function rdRender(){const host=$('#rdHost');if(!host.offsetParent)return;const C=colors(),l=10**+$('#rdL').value;$('#rdLv').textContent=l<0.1?l.toFixed(3):l.toFixed(2);
  const W=Math.max(240,host.clientWidth),H=Math.round(Math.min(300,W*.8)),ctx=fitCanvas($('#rdCv'),W,H),sc=Math.min(W,H)/4.2,ox=W*.3,oy=H*.75,sx=x=>ox+x*sc,sy=y=>oy-y*sc;
  ctx.strokeStyle=C.line;ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(0,oy);ctx.lineTo(W,oy);ctx.moveTo(ox,0);ctx.lineTo(ox,H);ctx.stroke();
  const[[a,b],[,d]]=RD.A,tr=(a+d)/2,q=Math.sqrt(((a-d)/2)**2+b*b),e1=tr+q,e2=tr-q,th=Math.atan2(e1-a,b);
  ctx.strokeStyle=C.muted;ctx.globalAlpha=.6;for(const lv of[0.1,0.4,1,2,3.5]){ctx.beginPath();for(let k=0;k<=80;k++){const t=k/80*2*Math.PI,u=Math.sqrt(2*lv/e1)*Math.cos(t),v=Math.sqrt(2*lv/e2)*Math.sin(t),x=RD.ws[0]+u*Math.cos(th)-v*Math.sin(th),y=RD.ws[1]+u*Math.sin(th)+v*Math.cos(th);k?ctx.lineTo(sx(x),sy(y)):ctx.moveTo(sx(x),sy(y))}ctx.stroke()}ctx.globalAlpha=1;
  ctx.strokeStyle=C.muted;ctx.lineWidth=2;ctx.beginPath();for(let k=0;k<=120;k++){const p=rdSolve(10**(-2+k/120*4.5));k?ctx.lineTo(sx(p[0]),sy(p[1])):ctx.moveTo(sx(p[0]),sy(p[1]))}ctx.stroke();
  ctx.strokeStyle=C.ink;ctx.lineWidth=2;const X=sx(RD.ws[0]),Y=sy(RD.ws[1]);ctx.beginPath();ctx.moveTo(X-5,Y-5);ctx.lineTo(X+5,Y+5);ctx.moveTo(X+5,Y-5);ctx.lineTo(X-5,Y+5);ctx.stroke();
  const w=rdSolve(l);ctx.fillStyle=C.accent;ctx.strokeStyle=C.surface;ctx.beginPath();ctx.arc(sx(w[0]),sy(w[1]),6,0,7);ctx.fill();ctx.stroke();
  ctx.font=bodyFont(12);ctx.fillStyle=C.ink;ctx.fillText('w₁',W-20,oy-6);ctx.fillText('w₂',ox+6,14);ctx.fillText('原点',ox+4,oy+14);
  const n0=Math.hypot(...RD.ws),n1=Math.hypot(...w),L=x=>{const u=x[0]-RD.ws[0],v=x[1]-RD.ws[1];return(a*u*u+2*b*u*v+d*v*v)/2};
  $('#rdCalc').textContent=`λ = ${l.toPrecision(3)}\n\n不加惩罚的最优解 w* = (${RD.ws[0].toFixed(2)}, ${RD.ws[1].toFixed(2)})  ‖w*‖ = ${n0.toFixed(3)}\n加惩罚后的最优解 w  = (${w[0].toFixed(3)}, ${w[1].toFixed(3)})  ‖w‖ = ${n1.toFixed(3)}\n\n原来的损失 L(w)      = ${L(w).toFixed(4)}（比最小值 0 高了）\n惩罚项 (λ/2)‖w‖²    = ${(l/2*n1*n1).toFixed(4)}\n两项加起来是最小的。\n\n${l<0.05?'λ 很小：几乎没有约束，w ≈ w*。':l>20?'λ 很大：w 被压得接近 0，模型几乎什么都学不到（欠拟合）。':'λ 适中：用一点训练损失换更小的权重。'}`}
$('#rdL').addEventListener('input',rdRender);
// ---------- 第 9 天：Dropout 丢掉了谁 ----------
const DP={h:[1.2,0.4,2.5,0.9,1.8,0.0,3.1,0.7,1.5,2.2,0.3,1.1,2.8,0.6,1.9,1.0],mask:null};
function dpSample(){const p=+$('#dpP').value;DP.mask=DP.h.map(()=>Math.random()>p);DP.avg=null;dpRender()}
function dpAvg(){const p=+$('#dpP').value,s=DP.h.map(()=>0);for(let r=0;r<1000;r++)DP.h.forEach((v,i)=>{if(Math.random()>p)s[i]+=p<1?v/(1-p):0});DP.avg=s.map(v=>v/1000);dpRender()}
function dpRender(){const p=+$('#dpP').value;$('#dpPv').textContent=p.toFixed(2);if(!DP.mask)DP.mask=DP.h.map((_,i)=>i%2===0);const k=1/(1-p);
  $('#dpGrid').innerHTML=DP.h.map((v,i)=>{const on=DP.mask[i];return`<div class="${on?'on':'off'}"><span class="small muted">原值 ${v.toFixed(1)}</span><b>${on?(v*k).toFixed(2):'0'}</b>${DP.avg?`<span class="small">平均 ${DP.avg[i].toFixed(2)}</span>`:''}</div>`}).join('');
  const kept=DP.mask.filter(Boolean).length,sum0=DP.h.reduce((a,b)=>a+b,0),sum1=DP.h.reduce((a,v,i)=>a+(DP.mask[i]?v*k:0),0);
  $('#dpCalc').textContent=`p = ${p.toFixed(2)}：每个单元以 ${p.toFixed(2)} 的概率被置 0，留下的乘 1/(1 − p) = ${k.toFixed(3)}\n这次留下 ${kept} / 16 个\n\n原来 16 个数的和：${sum0.toFixed(2)}\n丢弃并放大后的和：${sum1.toFixed(2)}（每次都不同，但期望等于原来的和）${DP.avg?`\n1000 次平均后的和：${DP.avg.reduce((a,b)=>a+b,0).toFixed(2)}，每格的平均都接近原值`:''}\n\nmodel.eval() 时 Dropout 什么也不做，输出就是原来的 h。`}
$('#dpP').addEventListener('input',()=>{DP.avg=null;dpRender()});$('#dpGo').onclick=dpSample;$('#dp100').onclick=dpAvg;
