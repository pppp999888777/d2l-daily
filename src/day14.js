// ---------- 第 14 天：门控怎样留住记忆 ----------
function gtRender(){const host=$('#gtHost');if(!host.offsetParent)return;const F=+$('#gtF').value,I=+$('#gtI').value,mode=$('#gtM').value;$('#gtFv').textContent=F.toFixed(2);$('#gtIv').textContent=I.toFixed(2);
  const T=40,noise=Array.from({length:T},(_,t)=>Math.sin(t*2.3)*0.8+Math.cos(t*5.1)*0.4);let c=1,cn=0;const keep=[1],grad=[1];let g=1,h=1,hn=0;
  for(let t=1;t<T;t++){if(mode==='lstm'){c=F*c;cn=F*cn+I*Math.tanh(noise[t]);keep.push(Math.abs(c));g*=F;grad.push(Math.abs(g))}
    else{const hp=Math.tanh(0.9*h+noise[t]),hq=Math.tanh(0.9*hn+noise[t]);g*=0.9*(1-hp*hp);h=hp;hn=hq;keep.push(Math.abs(h-hn));grad.push(Math.abs(g))}}
  const xs=Array.from({length:T},(_,i)=>i+1);host.innerHTML='<canvas id="gtCv" aria-label="记忆随时间的变化"></canvas>';
  drawPlot($('#gtCv'),{series:[{x:xs,y:keep.map(v=>Math.max(v,1e-12)),label:'第 1 步的信息还剩多少'},{x:xs,y:grad.map(v=>Math.max(v,1e-30)),label:'梯度 ∂(第 t 步)/∂(第 1 步)'}],hlines:[],xlabel:'时间步 t',ylabel:'',legend:true,yscale:'log',ylim:[1e-12,2]});
  $('#gtCalc').textContent=mode==='lstm'?`记忆元：Cₜ = F·C_{t−1} + I·C̃ₜ\n第 1 步写入的信息每一步乘 F，40 步后还剩 F³⁹ = ${(F**39).toExponential(2)}\n梯度 ∂C₄₀/∂C₁ = F³⁹，也是 ${(F**39).toExponential(2)}（所以图上两条线重合）\n\n${F>0.98?'遗忘门接近 1：信息和梯度都几乎原样穿过 40 步。':F<0.6?'遗忘门小：很快就忘了，梯度也消失。':'遗忘门适中：慢慢忘掉。'}\n输入门 I 只影响新写入的噪声有多少，不影响旧信息能留多久。\n真实 LSTM 里 F 每一步都是学出来的：需要记住时接近 1，需要清空时接近 0。`:`普通 RNN：hₜ = tanh(0.9·h_{t−1} + xₜ)\n每一步梯度乘 0.9 × tanh′，40 步后约 ${grad[T-1].toExponential(2)}\n第 1 步的不同（h₁ = 1 和 h₁ = 0 两条轨迹的差）也很快被噪声淹没。\n切回 LSTM，把遗忘门调到 0.99 对比一下。`}
['#gtF','#gtI'].forEach(s=>$(s).addEventListener('input',gtRender));$('#gtM').onchange=gtRender;
