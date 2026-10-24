// ---------- 第 13 天：梯度裁剪 ----------
function clRender(){const host=$('#clHost');if(!host.offsetParent)return;const C=colors(),L=+$('#clLen').value,ang=+$('#clAng').value*Math.PI/180,th=+$('#clTh').value;$('#clLenv').textContent=L.toFixed(1);$('#clAngv').textContent=$('#clAng').value+'°';$('#clThv').textContent=th;
  const g=[L*Math.cos(ang),L*Math.sin(ang)],k=Math.min(1,th/L),gc=[g[0]*k,g[1]*k];
  const W=Math.max(220,host.clientWidth),H=Math.min(W,300),ctx=fitCanvas($('#clCv'),W,H),cx=W/2,cy=H/2,sc=Math.min(W,H)/2/(Math.max(L,th)*1.2);
  ctx.strokeStyle=C.line;ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(0,cy);ctx.lineTo(W,cy);ctx.moveTo(cx,0);ctx.lineTo(cx,H);ctx.stroke();
  ctx.setLineDash([5,4]);ctx.strokeStyle=C.muted;ctx.beginPath();ctx.arc(cx,cy,th*sc,0,7);ctx.stroke();ctx.setLineDash([]);
  const arrow=(v,col,w)=>{const x=cx+v[0]*sc,y=cy-v[1]*sc,a=Math.atan2(cy-y,x-cx);ctx.strokeStyle=col;ctx.fillStyle=col;ctx.lineWidth=w;ctx.beginPath();ctx.moveTo(cx,cy);ctx.lineTo(x,y);ctx.stroke();ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(x-9*Math.cos(a-0.4),y+9*Math.sin(a-0.4));ctx.lineTo(x-9*Math.cos(a+0.4),y+9*Math.sin(a+0.4));ctx.fill()};
  arrow(g,C.accent,2);if(k<1)arrow(gc,C.pos,3);
  $('#clCalc').textContent=`原始梯度 g = (${g[0].toFixed(2)}, ${g[1].toFixed(2)})\n‖g‖ = ${L.toFixed(2)}，上限 θ = ${th}\n\n${k<1?`‖g‖ > θ，乘以 θ/‖g‖ = ${k.toFixed(3)}\n裁剪后 = (${gc[0].toFixed(2)}, ${gc[1].toFixed(2)})，长度正好是 ${th}`:'‖g‖ ≤ θ，不用裁剪，原样使用'}\n\n方向不变：裁剪前后两个箭头在同一条线上。\n真实模型里 g 是所有参数的梯度拼成的长向量，\n你论文的 DKT 大约有 7 万维，θ = 5。`}
['#clLen','#clAng','#clTh'].forEach(s=>$(s).addEventListener('input',clRender));
// ---------- 第 13 天：梯度沿时间往回传 ----------
function btRender(){const host=$('#btHost');if(!host.offsetParent)return;const w=+$('#btW').value,tr=+$('#btTr').value,act=$('#btAct').value;$('#btWv').textContent=w.toFixed(2);
  const T=60,xs=Array.from({length:T},(_,t)=>0.3*Math.sin(1.7*t)+0.2*Math.cos(0.9*t)),h=[0];for(let t=0;t<T;t++){const pre=w*h[t]+xs[t];h.push(act==='tanh'?Math.tanh(pre):pre)}
  const g=new Array(T+1).fill(0);g[T]=1;for(let t=T;t>=1;t--){const d=act==='tanh'?(1-h[t]*h[t]):1;g[t-1]=g[t]*w*d;if(tr&&T-(t-1)>tr)g[t-1]=0}
  const steps=Array.from({length:T},(_,i)=>i+1),vals=steps.map(i=>Math.max(1e-300,Math.abs(g[i])));
  host.innerHTML='<canvas id="btCv" aria-label="每个时间步的梯度大小"></canvas>';
  drawPlot($('#btCv'),{series:[{x:steps,y:steps.map(i=>tr&&T-i>=tr?NaN:vals[i-1]),label:'|∂L/∂hᵢ|'}],hlines:[],xlabel:'时间步 i（第 60 步是损失所在的位置）',ylabel:'',legend:false,yscale:'log'});
  const g1=Math.abs(g[1]),g30=Math.abs(g[30]);
  $('#btCalc').textContent=`每往回一步，梯度乘上 w × tanh′(·)${act==='lin'?'（线性时就是乘 w）':''}\n第 60 步：1\n第 30 步：${tr&&T-30>=tr?'被截断，为 0':g30.toExponential(2)}\n第 1 步：${tr&&T-1>=tr?'被截断，为 0':g1.toExponential(2)}\n\n${act==='lin'?(w<1?'w < 1：梯度按 w 的幂次衰减，远处几乎为 0（梯度消失）。':w>1?'w > 1：梯度按 w 的幂次放大（梯度爆炸），这时就需要梯度裁剪。':'w = 1：梯度不变。'):(w<1?'w < 1：梯度消失。':'tanh 会饱和（导数小于 1），把 w 放大的效果抵消掉一部分；w 很大时 h 贴近 ±1，导数接近 0，梯度反而又消失了。')}${tr?`\n截断：只往回传 ${tr} 步，更早的梯度直接是 0。`:''}`}
$('#btW').addEventListener('input',btRender);['#btTr','#btAct'].forEach(s=>$(s).onchange=btRender);
