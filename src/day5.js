
// ---------- 第 5 天：房价模型 ----------
const HOUSES=[[1400,10],[2000,3],[850,40],[3100,15]];
function houseRender(){const wa=+$('#hsWa').value,wg=+$('#hsWg').value,b=+$('#hsB').value;$('#hsWav').textContent=wa;$('#hsWgv').textContent=wg;$('#hsBv').textContent=b;
  const fmtUSD=v=>(v<0?'-':'')+'$'+Math.abs(Math.round(v)).toLocaleString();
  $('#hsCalc').textContent=`ŷ = Xw + b，X 的形状 (4, 2)，w 的形状 (2,)，ŷ 的形状 (4,)\n\n面积   房龄   预测价格 = ${wa}×面积 + (${wg})×房龄 + ${b}\n`+HOUSES.map(([ar,ag])=>`${String(ar).padStart(5)}  ${String(ag).padStart(4)}   ${fmtUSD(wa*ar+wg*ag+b).padStart(10)}`).join('\n')}
['#hsWa','#hsWg','#hsB'].forEach(s=>$(s).addEventListener('input',houseRender));

// ---------- 第 5 天：小批量 SGD ----------
const sgd={};
function sgdData(){let s=7;const r=()=>{s=(s*1664525+1013904223)>>>0;return s/4294967296},g=()=>{let u=0;while(!u)u=r();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*r())};
  sgd.X=[];sgd.y=[];for(let i=0;i<1000;i++){const a=g(),b=g();sgd.X.push([a,b]);sgd.y.push(2*a-3.4*b+4.2+0.01*g())}sgd.rng=r;
  let s11=0,s12=0,s22=0;for(const[a,b]of sgd.X){s11+=a*a;s12+=a*b;s22+=b*b}sgd.C=[s11/1000,s12/1000,s22/1000];sgdReset()}
function sgdFullLoss(w1,w2,b){let L=0;for(let i=0;i<1000;i++){const e=sgd.X[i][0]*w1+sgd.X[i][1]*w2+b-sgd.y[i];L+=e*e}return L/2000}
function sgdReset(){sgd.p=[0,0,0];sgd.path=[[0,0]];sgd.losses=[sgdFullLoss(0,0,0)];sgd.steps=0;sgd.epochs=0;sgdRender()}
function sgdTrain(){const bs=+$('#sgB').value,lr=10**+$('#sgLr').value,E=+$('#sgE').value;
  for(let e=0;e<E;e++){const idx=[...Array(1000).keys()];for(let i=999;i>0;i--){const j=Math.floor(sgd.rng()*(i+1));[idx[i],idx[j]]=[idx[j],idx[i]]}
    for(let s=0;s<1000;s+=bs){let g1=0,g2=0,gb=0;const n=Math.min(bs,1000-s);for(let k=s;k<s+n;k++){const[a,b]=sgd.X[idx[k]],er=a*sgd.p[0]+b*sgd.p[1]+sgd.p[2]-sgd.y[idx[k]];g1+=er*a;g2+=er*b;gb+=er}
      sgd.p=[sgd.p[0]-lr*g1/n,sgd.p[1]-lr*g2/n,sgd.p[2]-lr*gb/n];sgd.steps++;if(!sgd.p.every(Number.isFinite)||Math.abs(sgd.p[0])>1e6)break;
      if(sgd.path.length<6000)sgd.path.push([sgd.p[0],sgd.p[1]]);if(sgd.steps%Math.max(1,Math.floor(1000/bs/50))===0||bs>=100)sgd.losses.push(sgdFullLoss(...sgd.p))}sgd.epochs++}sgdRender()}
function sgdRender(){const host=$('#sgHost');if(!host.offsetParent)return;const C=colors(),cs=getComputedStyle(document.documentElement);const lr=10**+$('#sgLr').value;$('#sgLrv').textContent=lr.toFixed(3);$('#sgEv').textContent=$('#sgE').value;
  const W=Math.max(240,Math.min(360,host.clientWidth)),H=W,ctx=fitCanvas($('#sgCv'),W,H),X0=-1,X1=4,Y0=-5,Y1=0.5;const sx=x=>(x-X0)/(X1-X0)*W,sy=y=>(Y1-y)/(Y1-Y0)*H;
  const[c11,c12,c22]=sgd.C,q=(w1,w2)=>{const d1=w1-2,d2=w2+3.4;return.5*(c11*d1*d1+2*c12*d1*d2+c22*d2*d2)};const N=50,cw=W/N;
  for(let i=0;i<N;i++)for(let j=0;j<N;j++){const w1=X0+(i+.5)/N*(X1-X0),w2=Y1-(j+.5)/N*(Y1-Y0),v=Math.min(1,Math.sqrt(q(w1,w2))/3.5);ctx.fillStyle=`color-mix(in srgb, ${cs.getPropertyValue('--accent').trim()} ${Math.round(v*65)}%, ${C.surface})`;ctx.fillRect(i*cw,j*cw,cw+1,cw+1)}
  ctx.font=MONO;ctx.fillStyle=C.muted;ctx.textAlign='left';ctx.fillText('w₁ →',W-34,H-6);ctx.fillText('w₂ ↑',4,12);
  ctx.strokeStyle=C.ink;ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(sx(2)-6,sy(-3.4)-6);ctx.lineTo(sx(2)+6,sy(-3.4)+6);ctx.moveTo(sx(2)+6,sy(-3.4)-6);ctx.lineTo(sx(2)-6,sy(-3.4)+6);ctx.stroke();
  ctx.strokeStyle=C.pos;ctx.lineWidth=1.6;ctx.beginPath();sgd.path.forEach(([a,b],i)=>{const X=sx(Math.max(X0-1,Math.min(X1+1,a))),Y=sy(Math.max(Y0-1,Math.min(Y1+1,b)));i?ctx.lineTo(X,Y):ctx.moveTo(X,Y)});ctx.stroke();
  const[pa,pb]=sgd.path[sgd.path.length-1];ctx.fillStyle=C.pos;ctx.beginPath();ctx.arc(sx(pa),sy(pb),5,0,7);ctx.fill();
  const cv2=$('#sgLoss'),W2=Math.max(220,cv2.parentElement.clientWidth),H2=150,c2=fitCanvas(cv2,W2,H2),pl=40,pr=8,pt=8,pb2=20;const Ls=sgd.losses.filter(Number.isFinite).map(v=>Math.log10(Math.max(v,1e-6)));
  if(Ls.length>1){let lo=Math.floor(Math.min(...Ls)),hi=Math.ceil(Math.max(...Ls));if(hi===lo)hi++;const X=i=>pl+i/(Ls.length-1)*(W2-pl-pr),Y=v=>pt+(hi-v)/(hi-lo)*(H2-pt-pb2);
    c2.font=MONO;c2.fillStyle=C.muted;c2.strokeStyle=C.line;c2.textAlign='right';c2.textBaseline='middle';for(let k=lo;k<=hi;k++){c2.beginPath();c2.moveTo(pl,Y(k));c2.lineTo(W2-pr,Y(k));c2.stroke();c2.fillText(k>=0?String(10**k):(10**k).toFixed(-k),pl-4,Y(k))}
    c2.strokeStyle=C.accent;c2.lineWidth=1.8;c2.beginPath();Ls.forEach((v,i)=>i?c2.lineTo(X(i),Y(v)):c2.moveTo(X(i),Y(v)));c2.stroke();c2.textAlign='left';c2.textBaseline='top';c2.fillText('训练损失（对数刻度）',pl+4,pt)}
  const bs=+$('#sgB').value,L=sgd.losses[sgd.losses.length-1];
  $('#sgCalc').textContent=`批量大小 ${bs}，每个 epoch 更新 ${Math.ceil(1000/bs)} 次\n已训练 ${sgd.epochs} 个 epoch，共 ${sgd.steps} 次更新\n\nw = [${sgd.p[0].toFixed(4)}, ${sgd.p[1].toFixed(4)}]（真实 [2, -3.4]）\nb = ${sgd.p[2].toFixed(4)}（真实 4.2）\n训练损失 = ${Number.isFinite(L)?L.toExponential(3):'nan（发散了）'}${!sgd.p.every(Number.isFinite)||Math.abs(sgd.p[0])>1e5?'\n\n学习率太大，参数发散了。调小学习率再点“参数归零”。':''}`}
$('#sgRun').onclick=sgdTrain;$('#sgReset').onclick=sgdReset;$('#sgLr').oninput=sgdRender;$('#sgE').oninput=sgdRender;$('#sgB').onchange=sgdRender;
