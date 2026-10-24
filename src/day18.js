// ---------- 第 18 天：三种优化器赛跑 ----------
function opRun(kind,eta,beta){let x=[-5,-2],v=[0,0],s=[0,0];const g=p=>[0.2*p[0],4*p[1]],path=[x.slice()];
  for(let t=1;t<=40;t++){const gr=g(x);if(kind==='sgd')x=[x[0]-eta*gr[0],x[1]-eta*gr[1]];
    else if(kind==='mom'){v=[beta*v[0]+gr[0],beta*v[1]+gr[1]];x=[x[0]-eta*v[0],x[1]-eta*v[1]]}
    else{v=v.map((vi,i)=>0.9*vi+0.1*gr[i]);s=s.map((si,i)=>0.999*si+0.001*gr[i]**2);const vh=v.map(vi=>vi/(1-0.9**t)),sh=s.map(si=>si/(1-0.999**t));x=x.map((xi,i)=>xi-eta*vh[i]/(Math.sqrt(sh[i])+1e-8))}
    path.push(x.slice());if(!x.every(Number.isFinite)||Math.abs(x[1])>1e6)break}return path}
function opRender(){const host=$('#opHost');if(!host.offsetParent)return;const C=colors(),eta=10**+$('#opLr').value,beta=+$('#opB').value;$('#opLrv').textContent=eta.toFixed(3);$('#opBv').textContent=beta.toFixed(2);
  const W=Math.max(260,host.clientWidth),H=Math.round(Math.min(300,W*.62)),ctx=fitCanvas($('#opCv'),W,H),x0=-5.6,x1=1.2,y0=-3,y1=3,sx=x=>(x-x0)/(x1-x0)*W,sy=y=>H-(y-y0)/(y1-y0)*H;
  ctx.strokeStyle=C.line;ctx.lineWidth=1;for(const L of[0.05,0.3,1,2.5,5,8]){ctx.beginPath();for(let k=0;k<=120;k++){const a=k/120*2*Math.PI,xx=Math.sqrt(L/0.1)*Math.cos(a),yy=Math.sqrt(L/2)*Math.sin(a);k?ctx.lineTo(sx(xx),sy(yy)):ctx.moveTo(sx(xx),sy(yy))}ctx.stroke()}
  ctx.fillStyle=C.ink;ctx.beginPath();ctx.arc(sx(0),sy(0),3,0,7);ctx.fill();
  const runs=[['sgd','梯度下降',C.neg,[]],['mom','动量法',C.accent,[6,4]],['adam','Adam',C.pos,[2,3]]],res={};
  ctx.save();ctx.beginPath();ctx.rect(0,0,W,H);ctx.clip();
  runs.forEach(([k,n,col,d])=>{const p=opRun(k,eta,beta);res[k]=p;ctx.strokeStyle=col;ctx.fillStyle=col;ctx.lineWidth=2;ctx.setLineDash(d);ctx.beginPath();p.forEach((q,i)=>{const X=sx(Math.max(-1e4,Math.min(1e4,q[0]))),Y=sy(Math.max(-1e4,Math.min(1e4,q[1])));i?ctx.lineTo(X,Y):ctx.moveTo(X,Y)});ctx.stroke();ctx.setLineDash([]);p.forEach(q=>{ctx.beginPath();ctx.arc(sx(q[0]),sy(q[1]),2,0,7);ctx.fill()})});ctx.restore();
  ctx.font=bodyFont(12);runs.forEach(([k,n,col,d],i)=>{ctx.strokeStyle=col;ctx.setLineDash(d);ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(8,14+i*16);ctx.lineTo(28,14+i*16);ctx.stroke();ctx.setLineDash([]);ctx.fillStyle=C.ink;ctx.fillText(n,34,18+i*16)});
  const f=q=>0.1*q[0]**2+2*q[1]**2,last=k=>res[k][res[k].length-1],desc=k=>{const v=f(last(k));return!Number.isFinite(v)||v>1e6?'发散了':v.toExponential(2)};
  $('#opCalc').textContent=`学习率 η = ${eta.toFixed(3)}，动量 β = ${beta.toFixed(2)}；40 步后的函数值（越小越好）：\n梯度下降 ${desc('sgd')}\n动量法   ${desc('mom')}\nAdam     ${desc('adam')}\n\nx₂ 方向的梯度是 4x₂，梯度下降时 η > 0.5 就会在 x₂ 方向发散（每步乘 1 − 4η）。\nAdam 每一步对每个坐标走的距离大约都是 η（梯度被 √ŝ 归一化了），\n所以不管哪个方向陡、哪个方向平，它都差不多匀速前进；η 太大时会在最优点附近来回跳。`}
['#opLr','#opB'].forEach(s=>$(s).addEventListener('input',opRender));
