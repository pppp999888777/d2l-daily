// ---------- 第 16 天：位置编码的相似度 ----------
function peVec(i,d){const v=new Float64Array(d);for(let j=0;j<d;j+=2){const x=i/Math.pow(10000,j/d);v[j]=Math.sin(x);if(j+1<d)v[j+1]=Math.cos(x)}return v}
function peRender(){const host=$('#peHost');if(!host.offsetParent)return;const i=+$('#peI').value,d=+$('#peD').value;$('#peIv').textContent=i;
  const pi=peVec(i,d),xs=Array.from({length:60},(_,k)=>k),sim=xs.map(k=>{const pk=peVec(k,d);let s=0;for(let j=0;j<d;j++)s+=pi[j]*pk[j];return s/d*2});
  host.innerHTML='<canvas id="peCv" aria-label="位置相似度"></canvas>';
  drawPlot($('#peCv'),{series:[{x:xs,y:sim,label:`位置 ${i} 和各位置的相似度`}],hlines:[],xlabel:'位置',ylabel:'',legend:true});
  const pair=(a,b)=>{const pa=peVec(a,d),pb=peVec(b,d);let s=0;for(let j=0;j<d;j++)s+=pa[j]*pb[j];return(s/d*2).toFixed(4)};
  $('#peCalc').textContent=`相似度 = 两个位置编码的点积 × 2 / d（自己和自己是 1）\n\n位置 ${i} 和 ${Math.min(59,i+3)}：${pair(i,Math.min(59,i+3))}\n位置 0 和 3：${pair(0,3)}\n位置 40 和 43：${pair(40,43)}\n\n只要隔得一样远，相似度就一样：点积只和“相隔多远”有关，和绝对位置无关。\n离得越近越相似，所以模型能从位置编码里读出“这两道题隔了多远”。${d===8?'\n维度少时，远处的相似度会上下波动得比较厉害。':''}`}
$('#peI').addEventListener('input',peRender);$('#peD').onchange=peRender;
// ---------- 第 16 天：SAKT 预测下一题时在看什么 ----------
const SK={sk:[0,1,0,2,2,1,0,2,1,2],rs:[1,0,1,0,0,1,1,1,0,1],names:['A','B','C']};
function skRender(){const host=$('#skHost');if(!host.offsetParent)return;const C=colors(),q=+$('#skQ').value,r=+$('#skR').value,masked=$('#skM').value==='1';$('#skRv').textContent=r.toFixed(2);
  const T=SK.sk.length,tq=6;// 站在第 7 题（下标 6）之后，预测第 8 题；去掉掩码时会看到第 8 到 10 题
  const sc=SK.sk.map((s,j)=>(s===q?2:0)-r*Math.abs(tq-j)),use=sc.map((_,j)=>!masked||j<=tq),m=Math.max(...sc.filter((_,j)=>use[j])),e=sc.map((v,j)=>use[j]?Math.exp(v-m):0),z=e.reduce((a,b)=>a+b,0),w=e.map(v=>v/z),p=w.reduce((a,v,j)=>a+v*SK.rs[j],0);
  const W=Math.max(260,host.clientWidth),H=220,ctx=fitCanvas($('#skCv'),W,H),pl=8,cw=(W-2*pl)/T;ctx.font=bodyFont(12);ctx.textAlign='center';
  SK.sk.forEach((s,j)=>{const x=pl+j*cw,fut=j>tq;ctx.fillStyle=SK.rs[j]?C.pos:C.neg;ctx.globalAlpha=fut?.35:.8;ctx.fillRect(x+3,6,cw-6,34);ctx.globalAlpha=1;ctx.fillStyle=C.ink;ctx.fillText(SK.names[s],x+cw/2,22);ctx.fillText(SK.rs[j]?'对':'错',x+cw/2,36);
    const h=w[j]*(H-90);ctx.fillStyle=C.accent;ctx.fillRect(x+cw*0.25,H-28-h,cw*0.5,h);ctx.fillStyle=C.muted;ctx.fillText(w[j]>0.005?w[j].toFixed(2):(use[j]?'0':'×'),x+cw/2,H-32-h);ctx.fillText(String(j+1),x+cw/2,H-10)});
  ctx.strokeStyle=C.ink;ctx.setLineDash([4,4]);ctx.beginPath();const xb=pl+(tq+1)*cw;ctx.moveTo(xb,0);ctx.lineTo(xb,H);ctx.stroke();ctx.setLineDash([]);ctx.textAlign='left';ctx.fillStyle=C.muted;ctx.fillText('← 已经做过　未来 →',Math.max(4,xb-62),56);
  const fut=w.slice(tq+1).reduce((a,b)=>a+b,0);
  $('#skCalc').textContent=`现在做完了第 1 到 7 题，要预测第 8 题（知识点 ${SK.names[q]}）答对的概率。\n打分 = (同一个知识点 ? 2 : 0) − ${r.toFixed(2)} × 相隔的步数\n\n预测答对概率 = Σ 权重 × 对错 = ${p.toFixed(3)}\n${masked?'有因果掩码：第 8 到 10 题的权重是 0（图上标 ×）。':`去掉掩码：${(fut*100).toFixed(0)}% 的注意力落在了未来的题上，模型在“偷看答案”。训练时这样做损失会很低，但上线时根本拿不到未来的数据。`}\n${r<0.05?'不看远近：同知识点的题平分权重。':r>1?'只看最近的一两题，几乎不管知识点。':'同一个知识点、离得近的题权重最大。'}`}
['#skQ','#skM'].forEach(s=>$(s).onchange=skRender);$('#skR').addEventListener('input',skRender);
