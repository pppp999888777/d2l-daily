// 第 8 章用的时光机器文本：放进模拟文件系统
{const tm=document.getElementById('tmData');if(tm)MT.setStaticFile('../data/timemachine.txt',tm.textContent)}
{const af=document.getElementById('afData');if(af)MT.setStaticFile('../data/airfoil_self_noise.dat',af.textContent)}
// ============ 页面逻辑 ============
const $=s=>document.querySelector(s),$$=(s,r=document)=>[...r.querySelectorAll(s)];
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const store={get(k,d){try{const v=localStorage.getItem('d2l:'+k);return v==null?d:JSON.parse(v)}catch(e){return d}},set(k,v){try{localStorage.setItem('d2l:'+k,JSON.stringify(v))}catch(e){}}};
function colors(){const cs=getComputedStyle(document.documentElement),g=n=>cs.getPropertyValue(n).trim();const o={};for(const k of['bg','surface','sunk','ink','muted','line','accent','pos','neg'])o[k]=g('--'+k);return o}
function fitCanvas(cv,W,H){const r=window.devicePixelRatio||1;cv.width=Math.round(W*r);cv.height=Math.round(H*r);cv.style.width=W+'px';cv.style.height=H+'px';const c=cv.getContext('2d');c.setTransform(r,0,0,r,0,0);return c}
const MONO='11px "JetBrains Mono",ui-monospace,monospace';
const bodyFont=px=>`${px}px ${getComputedStyle(document.body).fontFamily}`;
const DAYS=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'];

// ---------- 张量图示 ----------
function cellText(v,dtype){if(dtype==='bool')return v?'T':'F';if(dtype==='int64')return String(v);if(!Number.isFinite(v))return String(v);return Number.isInteger(v)?String(v):(Math.abs(v)>=100?v.toFixed(0):v.toFixed(2))}
function cellStyle(v,dtype,mx){if(dtype==='bool')return v?'background:color-mix(in srgb,var(--accent) 38%,var(--surface))':'';if(!Number.isFinite(v)||!mx||v===0)return'';
  const t=Math.round(Math.min(1,Math.abs(v)/mx)**0.8*55);return`background:color-mix(in srgb,var(${v>0?'--pos':'--neg'}) ${t}%,var(--surface))`}
const GROUPC=['--accent','--pos','--neg'];
function gridHTML(shape,data,dtype,cls=()=>'',mxOverride,styleFn){const nd=shape.length,st=MT.strides(shape);let mx=mxOverride;if(mx==null){mx=0;for(const v of data)if(Number.isFinite(v)&&Math.abs(v)>mx)mx=Math.abs(v)}
  const td=off=>`<td class="${cls(off)}" style="${styleFn?styleFn(off):cellStyle(data[off],dtype,mx)}" title="${esc(cellText(data[off],dtype))}">${esc(cellText(data[off],dtype))}</td>`;
  const mat=(rows,cols,base,rs,colStride,showIdx)=>{let h='<table class="tm">';if(showIdx){h+='<tr><td class="i"></td>';for(let j=0;j<cols;j++)h+=`<td class="i">${j}</td>`;h+='</tr>'}
    for(let i=0;i<rows;i++){h+='<tr>'+(showIdx?`<td class="i">${i}</td>`:'');for(let j=0;j<cols;j++)h+=td(base+i*rs+j*colStride);h+='</tr>'}return h+'</table>'};
  if(nd===0)return`<div class="tg">${mat(1,1,0,0,0,false)}<span class="ax">0 维（标量）</span></div>`;
  if(nd===1)return`<div class="tg"><span class="ax">轴 0 →（长度 ${shape[0]}）</span>${mat(1,shape[0],0,0,1,true)}</div>`;
  if(nd===2)return`<div class="tg"><span class="ax">轴 1（列）→　　轴 0（行）↓</span>${mat(shape[0],shape[1],0,st[0],1,true)}</div>`;
  let h=`<div class="tg"><span class="ax">轴 0 有 ${shape[0]} 块；每块里 轴 1（行）↓ 轴 2（列）→</span><div class="grid3">`;
  for(let b=0;b<shape[0];b++)h+=`<div class="tg"><span class="ax">第 ${b} 块</span>${mat(shape[1],shape[2],b*st[0],st[1],1,false)}</div>`;return h+'</div></div>'}
const shapeStr=s=>'('+s.join(', ')+(s.length===1?',':'')+')';
function labelHTML(T,title){return`<span class="lab">${title?esc(title)+' · ':''}形状 <b>${shapeStr(T.shape)}</b> · ${T.data.length} 个元素 · ${T.dtype}${T.requires_grad?' · 需要梯度':''}</span>`}
function expandedHTML(S,out,mx){const n=out.reduce((a,b)=>a*b,1),k=out.length-S.shape.length,sst=MT.strides(S.shape),data=new Float64Array(n),copied=new Uint8Array(n),idx=new Array(out.length).fill(0);
  for(let f=0;f<n;f++){let off=0,cp=false;for(let d=0;d<out.length;d++){if(d<k){if(idx[d]>0)cp=true;continue}const sd=S.shape[d-k];if(sd===1){if(idx[d]>0)cp=true}else off+=idx[d]*sst[d-k]}data[f]=S.data[off];copied[f]=cp?1:0;
    for(let d=out.length-1;d>=0;d--){if(++idx[d]<out[d])break;idx[d]=0}}
  return gridHTML(out,data,S.dtype,off=>copied[off]?'cp':'',mx)}
function vizTensor(T,title){if(T.data.length>240||T.shape.length>3)return`<div class="item">${labelHTML(T,title)}<span class="small muted">元素太多，不画表格。形状 ${shapeStr(T.shape)} 的意思是：${T.shape.map((d,i)=>`轴 ${i} 长 ${d}`).join('，')}。</span></div>`;
  const m=T.meta;
  if(m&&m.kind==='bcast'){const mx=Math.max(...[...m.a.data,...m.b.data,...T.data].filter(Number.isFinite).map(Math.abs),0);
    return`<div class="item"><span class="lab">${esc(title||'')} 广播：${shapeStr(m.a.shape)} 和 ${shapeStr(m.b.shape)} 都扩展成 ${shapeStr(T.shape)}，虚线格是复制出来的</span><div class="vz">
      <div class="item">${expandedHTML(m.a,T.shape,mx)}<span class="lab">左边 ${shapeStr(m.a.shape)} → ${shapeStr(T.shape)}</span></div><span class="op">${esc(m.op)}</span>
      <div class="item">${expandedHTML(m.b,T.shape,mx)}<span class="lab">右边 ${shapeStr(m.b.shape)} → ${shapeStr(T.shape)}</span></div><span class="op">=</span>
      <div class="item">${gridHTML(T.shape,T.data,T.dtype,()=>'',mx)}${labelHTML(T)}</div></div></div>`}
  if(m&&m.kind==='index')return`<div class="item"><span class="lab">${esc(title||'')} 从原张量 ${shapeStr(m.src.shape)} 里选出粗框的格子</span><div class="vz">
      <div class="item">${gridHTML(m.src.shape,m.src.data,m.src.dtype,off=>m.mask[off]?'sel':'dim')}<span class="lab">原张量</span></div><span class="op">→</span>
      <div class="item">${gridHTML(T.shape,T.data,T.dtype)}${labelHTML(T)}</div></div></div>`;
  if(m&&m.kind==='reduce'&&m.src.shape.length===2&&m.dims.length===1){const S=m.src,ax=m.dims[0],[R,Cn]=S.shape;const grp=off=>ax===0?off%Cn:Math.floor(off/Cn);
    const sty=off=>`background:color-mix(in srgb,var(${GROUPC[grp(off)%3]}) ${18+(grp(off)%2)*14}%,var(--surface))`;const nm={sum:'求和',mean:'求平均',max:'取最大',min:'取最小'}[m.op];
    const outSty=i=>`background:color-mix(in srgb,var(${GROUPC[i%3]}) ${18+(i%2)*14}%,var(--surface))`;
    return`<div class="item"><span class="lab">${esc(title||'')} 沿轴 ${ax}${nm}：${ax===0?'每一列上下':'每一行左右'}的格子合成一个数，同色的格子归到一起。轴 ${ax} 的长度 ${S.shape[ax]} 从形状里${m.keep?'变成 1':'消失'}</span><div class="vz">
      <div class="item">${gridHTML(S.shape,S.data,S.dtype,()=>'',null,sty)}<span class="lab">原张量 ${shapeStr(S.shape)}</span></div><span class="op">→</span>
      <div class="item">${gridHTML(T.shape,T.data,T.dtype,()=>'',null,off=>outSty(off))}${labelHTML(T)}</div></div></div>`}
  if(m&&m.kind==='mm'){const A=m.a,Bm=m.b;return`<div class="item"><span class="lab">${esc(title||'')} 矩阵乘法：左边的每一行和右边的每一列做点积。${shapeStr(A.shape)} @ ${shapeStr(Bm.shape)} → ${shapeStr(T.shape)}</span><div class="vz">
      <div class="item">${gridHTML(A.shape,A.data,A.dtype)}<span class="lab">左 ${shapeStr(A.shape)}</span></div><span class="op">@</span><div class="item">${gridHTML(Bm.shape,Bm.data,Bm.dtype)}<span class="lab">右 ${shapeStr(Bm.shape)}</span></div><span class="op">=</span>
      <div class="item">${gridHTML(T.shape,T.data,T.dtype)}${labelHTML(T)}</div></div></div>`}
  if(m&&m.kind==='cat'){const p0=m.parts[0],dim=m.dim;const st=MT.strides(T.shape);const inFirst=off=>{const i=Math.floor(off/st[0]),j=off%st[0];return dim===0?i<p0.shape[0]:j<p0.shape[1]};
    return`<div class="item">${gridHTML(T.shape,T.data,T.dtype,off=>inFirst(off)?'':'p1')}${labelHTML(T,title)}<span class="legend"><span>没下划线的来自第 1 个张量</span><span><i style="background:var(--accent)"></i>带蓝色下划线的来自第 2 个</span><span>沿 dim=${dim}（轴 ${dim}）拼接</span></span></div>`}
  if(m&&m.kind==='assign')return`<div class="item">${gridHTML(T.shape,T.data,T.dtype,off=>m.mask[off]?'chg':'')}${labelHTML(T,title)}<span class="legend"><span><i style="background:var(--pos)"></i>橙色框是刚被改写的格子</span></span></div>`;
  return`<div class="item">${gridHTML(T.shape,T.data,T.dtype)}${labelHTML(T,title)}</div>`}
// 计算图：从结果往回追到叶子
function graphHTML(T,env){if(!T.grad_fn)return'';const names=new Map();for(const[k,v]of Object.entries(env.vars))if(MT.isT(v)&&!names.has(v))names.set(v,k);
  const seen=new Set();let count=0;
  const node=(t,depth)=>{if(count++>24)return'<li class="muted">……</li>';const nm=names.get(t);const sh=shapeStr(t.shape);
    if(!t.grad_fn){if(t.requires_grad)return`<li><span class="gn leaf">${esc(nm||'叶子')}</span> 叶子节点，需要梯度，形状 ${sh}${t.grad?`，现在的 .grad = <code>${esc(MT.repr(t.grad).replace(/^tensor\(|\)$/g,''))}</code>`:''}</li>`;
      return`<li><span class="gn const">${esc(nm||(t.weak?'常数 '+cellText(t.data[0],t.dtype):'常数张量'))}</span> 不需要梯度${t.weak?'':'，形状 '+sh}</li>`}
    if(seen.has(t))return`<li><span class="gn">${esc(nm||'中间结果')}</span>（上面已画过）</li>`;seen.add(t);
    return`<li><span class="gn op">${esc(nm||'中间结果')}</span> = <code>${esc(t.grad_fn.name)}</code> 算出来的，形状 ${sh}${t.grad_fn.freed?'，<span class="muted">这张图已经反向传播过</span>':''}<ul>${t.grad_fn.inputs.filter(MT.isT).map(i=>node(i,depth+1)).join('')}</ul></li>`};
  return`<div class="graph"><span class="lab">计算图：从结果往下追，<code>backward()</code> 时梯度沿着这条路往回传到叶子节点</span><ul>${node(T,0)}</ul></div>`}
function vizValue(v,env){if(!v)return'';if(MT.isT(v))return`<div class="vz">${vizTensor(v)}</div>`+graphHTML(v,env);
  if(v.t==='tuple'&&v.items.some(MT.isT))return`<div class="vz">${v.items.map((x,i)=>MT.isT(x)?vizTensor(x,`第 ${i+1} 个`):`<div class="item"><span class="lab">第 ${i+1} 个</span><code>${esc(MT.repr(x).slice(0,200))}</code></div>`).join('')}</div>`;
  if(v.t==='size')return`<div class="vz"><div class="item"><span class="lab">形状里有 ${v.dims.length} 个数，就是 ${v.dims.length} 个轴：${v.dims.map((d,i)=>`轴 ${i} 长 ${d}`).join('，')||'0 维（标量）'}</span></div></div>`;return''}

// ---------- 画图（d2l.plot / plt.plot） ----------
const SERIESC=['--accent','--pos','--neg','--muted'];
function drawPlot(cv,fig){const C=colors(),cs=getComputedStyle(document.documentElement);const W=Math.max(260,Math.min(560,cv.parentElement.clientWidth||520)),H=Math.round(W*0.62),ctx=fitCanvas(cv,W,H);
  // 坐标轴可以是线性或对数（xscale / yscale = 'log'），对数轴先取 log10 再按线性画
  const pl=50,pr=12,pt=fig.legend?30:12,pb=36,lx=fig.xscale==='log',ly=fig.yscale==='log',TX=v=>lx?(v>0?Math.log10(v):NaN):v,TY=v=>ly?(v>0?Math.log10(v):NaN):v;
  const cts=fig.contours||[];const xs=[...fig.series.flatMap(s=>s.x),...cts.flatMap(c=>[Math.min(...c.x),Math.max(...c.x)])].map(TX).filter(Number.isFinite),ys=[...fig.series.flatMap(s=>s.y),...fig.hlines,...cts.flatMap(c=>[Math.min(...c.y),Math.max(...c.y)])].map(TY).filter(Number.isFinite);if(!xs.length||!ys.length)return;
  let x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);if(x1===x0)x1=x0+1;if(y1===y0){y1+=1;y0-=1}const py=(y1-y0)*.06;y0-=py;y1+=py;
  if(fig.xlim){const[a,b]=fig.xlim.map(TX);if(Number.isFinite(a))x0=a;if(Number.isFinite(b))x1=b}if(fig.ylim){const[a,b]=fig.ylim.map(TY);if(Number.isFinite(a))y0=a;if(Number.isFinite(b))y1=b}
  const nice=(lo,hi,n)=>{const sp=hi-lo,st0=sp/n,mg=10**Math.floor(Math.log10(st0)),e=st0/mg,st=(e>=7.5?10:e>=3.5?5:e>=1.5?2:1)*mg;const t=[];for(let v=Math.ceil(lo/st)*st;v<=hi+st*1e-9;v+=st)t.push(+v.toFixed(10));return t};
  const SUP={'-':'⁻',0:'⁰',1:'¹',2:'²',3:'³',4:'⁴',5:'⁵',6:'⁶',7:'⁷',8:'⁸',9:'⁹'};
  const ticks=(lo,hi,n,lg)=>{if(!lg)return nice(lo,hi,n).map(t=>[t,String(+t.toPrecision(6))]);const p=nice(lo,hi,n+1).filter(Number.isInteger);if(p.length>=2)return p.map(t=>[t,'10'+String(t).replace(/./g,c=>SUP[c])]);
    const r=[];for(let k=Math.floor(lo);k<=Math.ceil(hi);k++)for(const m of[1,2,5]){const t=Math.log10(m)+k;if(t>=lo&&t<=hi)r.push([t,String(+(10**t).toPrecision(2))])}return r};
  const sx0=x=>pl+(x-x0)/(x1-x0)*(W-pl-pr),sy0=y=>pt+(y1-y)/(y1-y0)*(H-pt-pb),sx=x=>sx0(TX(x)),sy=y=>sy0(TY(y));
  ctx.font=MONO;ctx.fillStyle=C.muted;ctx.strokeStyle=C.line;ctx.lineWidth=1;ctx.textAlign='right';ctx.textBaseline='middle';
  for(const[t,lab]of ticks(y0,y1,4,ly)){const y=Math.round(sy0(t))+.5;ctx.beginPath();ctx.moveTo(pl,y);ctx.lineTo(W-pr,y);ctx.stroke();ctx.fillText(lab,pl-6,y)}
  ctx.textAlign='center';ctx.textBaseline='top';for(const[t,lab]of ticks(x0,x1,5,lx)){const x=Math.round(sx0(t))+.5;ctx.fillText(lab,x,H-pb+6)}
  ctx.font=bodyFont(12);ctx.fillStyle=C.ink;if(fig.xlabel)ctx.fillText(fig.xlabel,(pl+W-pr)/2,H-16);if(fig.ylabel){ctx.save();ctx.translate(13,(pt+H-pb)/2);ctx.rotate(-Math.PI/2);ctx.textBaseline='middle';ctx.fillText(fig.ylabel,0,0);ctx.restore()}
  for(const h of fig.hlines){ctx.strokeStyle=C.ink;ctx.setLineDash([5,4]);ctx.beginPath();ctx.moveTo(pl,sy(h));ctx.lineTo(W-pr,sy(h));ctx.stroke();ctx.setLineDash([])}
  ctx.save();ctx.beginPath();ctx.rect(pl,pt,W-pl-pr,H-pt-pb);ctx.clip();
  // 等高线：在网格上用 marching squares 找每个等值线经过的线段
  for(const c of cts){const{nx,ny,x,y,z}=c,zs=z.filter(Number.isFinite),z0=Math.min(...zs),z1=Math.max(...zs),lv=[];for(let k=1;k<=8;k++)lv.push(z0+(z1-z0)*(k/9)**1.6);
    ctx.strokeStyle=C.muted;ctx.lineWidth=1;ctx.globalAlpha=.7;const at=(i,j)=>i*ny+j;
    for(const L of lv){ctx.beginPath();for(let i=0;i<nx-1;i++)for(let j=0;j<ny-1;j++){const p=[[i,j],[i+1,j],[i+1,j+1],[i,j+1]],v=p.map(([a,b])=>z[at(a,b)]),pts=[];
      for(let e=0;e<4;e++){const a=e,b=(e+1)%4;if((v[a]<L)!==(v[b]<L)){const t=(L-v[a])/(v[b]-v[a]),pa=p[a],pb=p[b];pts.push([x[at(...pa)]+(x[at(...pb)]-x[at(...pa)])*t,y[at(...pa)]+(y[at(...pb)]-y[at(...pa)])*t])}}
      for(let k=0;k+1<pts.length;k+=2){ctx.moveTo(sx(pts[k][0]),sy(pts[k][1]));ctx.lineTo(sx(pts[k+1][0]),sy(pts[k+1][1]))}}ctx.stroke()}ctx.globalAlpha=1}
  const dashOf=(s,i)=>Array.isArray(s.dash)?s.dash:(i===1?[6,4]:i===2?[2,3]:i===3?[8,3,2,3]:i===4?[1,4]:[]);
  const HEX={'#ff7f0e':'--pos','#1f77b4':'--accent','#2ca02c':'--neg'},colOf=(s,i)=>s.hex?(HEX[s.hex]?cs.getPropertyValue(HEX[s.hex]).trim():s.hex):s.color?cs.getPropertyValue(s.color).trim():cs.getPropertyValue(SERIESC[i%4]).trim();
  fig.series.forEach((s,i)=>{const col=colOf(s,i);if(s.scatter){ctx.fillStyle=col;ctx.globalAlpha=.7;s.x.forEach((x,k)=>{ctx.beginPath();ctx.arc(sx(x),sy(s.y[k]),1.8,0,7);ctx.fill()});ctx.globalAlpha=1;return}
    ctx.strokeStyle=col;ctx.lineWidth=2;ctx.setLineDash(dashOf(s,i));ctx.beginPath();let pen=false;s.x.forEach((x,k)=>{const X=sx(x),Y=sy(s.y[k]);if(!Number.isFinite(X)||!Number.isFinite(Y)){pen=false;return}pen?ctx.lineTo(X,Y):ctx.moveTo(X,Y);pen=true});ctx.stroke();ctx.setLineDash([]);
    if(s.x.length<=30||s.marker){ctx.fillStyle=col;s.x.forEach((x,k)=>{const X=sx(x),Y=sy(s.y[k]);if(Number.isFinite(X)&&Number.isFinite(Y)){ctx.beginPath();ctx.arc(X,Y,2.5,0,7);ctx.fill()}})}});ctx.restore();
  if(fig.legend){let lx0=pl,row=0;ctx.font=bodyFont(12);ctx.textAlign='left';ctx.textBaseline='middle';fig.series.forEach((s,i)=>{if(!s.label)return;const w=ctx.measureText(s.label).width+44;if(lx0+w>W&&lx0>pl){lx0=pl;row++}const y=12+row*16;const col=colOf(s,i);ctx.strokeStyle=col;ctx.lineWidth=2;ctx.setLineDash(dashOf(s,i));ctx.beginPath();ctx.moveTo(lx0,y);ctx.lineTo(lx0+18,y);ctx.stroke();ctx.setLineDash([]);ctx.fillStyle=C.ink;ctx.fillText(s.label,lx0+22,y);lx0+=w})}
  cv._fig=fig}

// ---------- 代码格 ----------
const ENVS={},EXEC={},RAN={};
const dayOf=el=>el.closest('main.day').dataset.day;
const envOf=day=>ENVS[day]||(ENVS[day]=MT.newEnv());
const HINTS={NameError:'这个名字还没定义。通常是上面的格子还没运行：点这一天开头的“从头全部运行”，或者先运行定义它的那个格子。',
  SyntaxError:'写法有问题。检查括号是否成对、是不是用了中文标点、冒号后面有没有换行缩进。',IndentationError:'缩进不对。同一个代码块里的行要对齐，冒号后面的下一行要多缩进 4 个空格。',
  RuntimeError:'运算本身出错了，最常见的原因是形状对不上。先打印两边的 .shape 看看。',IndexError:'下标超出范围了。下标从 0 开始，长度为 n 的轴最大下标是 n-1。',
  TypeError:'类型不对，比如该给数字的地方给了文字。看报错说的是哪种类型。',AttributeError:'这个对象没有这个属性或方法，检查一下拼写。',ValueError:'值不合适，看看错误信息里说的是哪个值。',
  KeyError:'没有这个名字的列或键，检查一下拼写和大小写。',ModuleNotFoundError:'这个网页只装了课程用到的几个包：torch、numpy、pandas、os、math、random、d2l。'};
const CHECKS={
  h24:env=>{const h=env.vars.h;if(!h||!MT.isT(h))return['bad','还没有名为 h 的张量。'];if(h.shape.join()==='24,64')return['ok',`完成。h 的形状是 [24, 64]，一共 ${h.data.length} 个数。换一种方法再做一次试试。`];return['bad',`h 现在的形状是 [${h.shape.join(', ')}]，目标是 [24, 64]。`]},
  z100:env=>{const o=env.vars.one;if(!o||!MT.isT(o))return['bad','还没有名为 one 的张量。'];if(o.shape.join()==='100,64')return['ok','对，取出一个学生得到 [100, 64]。再试试 z[0, -1] 或 z[0][-1]，取第 1 个学生最后一步，形状是 [64]。'];if(o.shape.join()==='64')return['ok','对，这是第 1 个学生最后一步的 64 个数，形状 [64]。'];return['bad',`one 的形状是 [${o.shape.join(', ')}]。取第 1 个学生用 z[0]。`]},
  clean:env=>{const c=env.vars.clean;if(!c||c.t!=='df')return['bad','还没有名为 clean 的表格。'];if(c.cols.includes('Alley'))return['bad',`clean 里还有 Alley 列。现在的列是 ${c.cols.join('、')}。用 data.drop(columns=[worst])。`];return['ok',`完成。删掉了缺失最多的 Alley，剩下 ${c.cols.join('、')}。`]},
  T42:env=>{const T=env.vars.T;if(!T||!MT.isT(T))return['bad','还没有名为 T 的张量。'];if(T.shape.join()!=='4,2')return['bad',`T 的形状是 [${T.shape.join(', ')}]，应该是 [4, 2]。先做完练习 1。`];if([...T.data].some(Number.isNaN))return['bad','T 里还有 nan，先补缺失值。'];return['ok','完成。4 套房子、每套 2 个数（房间数和价格），没有缺失值。']}};
function renderResult(cell,r,env){const out=cell.querySelector('.out'),viz=cell.querySelector('.viz');
  if(r.ok){out.className='out';out.textContent=r.text;out.hidden=!r.text;viz.innerHTML=vizValue(r.value,env)}
  else{out.className='out err';out.textContent=r.text;const tip=document.createElement('span');tip.className='tip';tip.textContent=HINTS[r.errType]||'';if(tip.textContent)out.appendChild(tip);out.hidden=false;viz.innerHTML=''}
  for(const fig of r.plots||[]){const wrap=document.createElement('div');wrap.className='plot';viz.appendChild(wrap);if(fig.kind==='fig'){renderFig(wrap,fig);continue}const cv=document.createElement('canvas');cv.setAttribute('aria-label','代码画出的图');wrap.appendChild(cv);drawPlot(cv,fig)}
  viz.hidden=!viz.innerHTML}
function runCellEl(cell){const day=dayOf(cell),env=envOf(day),ta=cell.querySelector('textarea');const r=MT.runCell(ta.value,env);EXEC[day]=(EXEC[day]||0)+1;
  cell.querySelector('.cnt').textContent=`[${EXEC[day]}]`;cell.classList.remove('stale');renderResult(cell,r,env);
  const ck=cell.dataset.check;if(ck){const box=cell.querySelector('.check');const[k,msg]=r.ok?CHECKS[ck](env):['bad','先让代码不报错。'];box.innerHTML=`<span class="${k}">${k==='ok'?'✓ ':'· '}</span>${esc(msg)}`;box.hidden=false}
  return r.ok}
function autoSize(ta){if(!ta.offsetParent)return;ta.style.height='auto';ta.style.height=ta.scrollHeight+2+'px'}
function setupCells(){$$('.cell').forEach((cell,i)=>{const ta=cell.querySelector('textarea');ta.id='cell'+i;ta.spellcheck=false;ta.setAttribute('aria-label',`代码格 ${i+1}`);cell.dataset.orig=ta.value;ta.rows=Math.max(1,ta.value.split('\n').length);
  const bar=document.createElement('div');bar.className='bar';bar.innerHTML=`<span class="cnt">[ ]</span><span class="hint">Ctrl/⌘ + Enter 运行</span><span class="sp"></span><button class="btn sm" data-a="reset">还原</button><button class="btn sm primary" data-a="run">运行</button>`;
  cell.insertBefore(bar,ta);const out=document.createElement('pre');out.className='out';out.hidden=true;const viz=document.createElement('div');viz.className='viz';viz.hidden=true;cell.append(out,viz);
  if(cell.dataset.check){const c=document.createElement('div');c.className='check';c.hidden=true;cell.append(c)}
  bar.querySelector('[data-a=run]').onclick=()=>{if(cell.dataset.manual){const out=cell.querySelector('.out');out.className='out';out.textContent='正在运行，请稍等……';out.hidden=false;cell.querySelector('.viz').hidden=true;setTimeout(()=>runCellEl(cell),40)}else runCellEl(cell)};
  bar.querySelector('[data-a=reset]').onclick=()=>{ta.value=cell.dataset.orig;autoSize(ta);cell.classList.add('stale')};
  ta.addEventListener('input',()=>{autoSize(ta);cell.classList.add('stale')});
  ta.addEventListener('keydown',e=>{if((e.ctrlKey||e.metaKey)&&e.key==='Enter'){e.preventDefault();runCellEl(cell)}
    else if(e.key==='Tab'&&!e.shiftKey){e.preventDefault();const s=ta.selectionStart;ta.setRangeText('    ',s,ta.selectionEnd,'end');autoSize(ta)}})})}
function runAll(day){ENVS[day]=MT.newEnv();EXEC[day]=0;MT.resetFS();let errs=0;const main=$('#day'+day);
  $$('.cell',main).forEach(c=>{if(c.dataset.manual){const out=c.querySelector('.out');out.className='out';out.textContent='';const t=document.createElement('span');t.className='tip';t.textContent=`这一格要手动运行：${c.dataset.manual}。准备好了点这一格右上角的“运行”。`;out.appendChild(t);out.hidden=false;return}if(!runCellEl(c))errs++});RAN[day]=true;
  const expected={'1':0,'2':5,'3':3,'4':0,'5':0,'6':0,'7':0,'8':0,'9':0,'10':0,'11':2,'12':0,'13':0,'14':0,'15':0,'16':0,'17':0,'18':0,'19':0,'20':0}[day]||0;
  $$('.runAllMsg',main).forEach(m=>m.textContent=errs>expected?`运行完了，有 ${errs} 个格子报错，往下找红字。`:expected?`全部运行完了。其中 ${expected} 个格子是故意报错的，用来练习读报错。`:'全部运行完了，没有报错。')}
document.addEventListener('click',e=>{const b=e.target.closest('.runAll');if(b)runAll(b.dataset.day)});

// ---------- 第 1 天：修水管 ----------
const kn={b:0,w:0,steps:0,stage:0};
function kData(){return $('#kThird').checked?[[3,350],[2,250],[4,480]]:[[3,350],[2,250]]}
function kRender(){const D=kData(),b=kn.b,w=kn.w;$('#kB').value=Math.round(b);$('#kW').value=Math.round(w);$('#kBv').textContent=b.toFixed(1);$('#kWv').textContent=w.toFixed(1);
  let L=0;const rows=D.map(([h,y])=>{const p=b+w*h,e=p-y;L+=e*e;return`<tr><td class="num">${h} 小时</td><td class="num">${y}</td><td class="num">${p.toFixed(1)}</td><td class="num">${e>=0?'+':''}${e.toFixed(1)}</td></tr>`});L/=D.length;
  $('#kTbl').innerHTML=`<thead><tr><th>账单（特征）</th><th>实际金额（标签）</th><th>模型猜的 b + w×小时</th><th>差多少</th></tr></thead><tbody>${rows.join('')}</tbody>`;
  $$('#kSteps li').forEach((li,i)=>li.classList.toggle('on',i===kn.stage));
  $('#kCalc').textContent=`模型：金额 = ${b.toFixed(1)} + ${w.toFixed(1)} × 小时\n平方误差（平均）= ${L.toFixed(2)}\n已训练 ${kn.steps} 步${L<0.5&&!$('#kThird').checked?'\n\n找到了：上门费约 50 美元，每小时约 100 美元，和书里的答案一样。':''}${$('#kThird').checked&&kn.steps>500?'\n\n三张账单不可能全猜对，现在已经是误差最小的那条线了。':''}`}
function kTrain(n){const D=kData(),lr=0.07;for(let s=0;s<n;s++){let gb=0,gw=0;for(const[h,y]of D){const e=kn.b+kn.w*h-y;gb+=2*e;gw+=2*e*h}kn.b-=lr*gb/D.length;kn.w-=lr*gw/D.length;kn.steps++}}
$('#kTrain').onclick=()=>{kn.stage=1;kRender();setTimeout(()=>{kn.stage=2;kTrain(200);kRender();setTimeout(()=>{kn.stage=3;kRender()},350)},250)};
$('#kReset').onclick=()=>{kn.b=0;kn.w=0;kn.steps=0;kn.stage=0;kRender()};
$('#kB').oninput=()=>{kn.b=+$('#kB').value;kRender()};$('#kW').oninput=()=>{kn.w=+$('#kW').value;kRender()};$('#kThird').onchange=kRender;
function imRender(){const w=parseInt($('#imW').value)||0,h=parseInt($('#imH').value)||0,c=parseInt($('#imC').value)||0;
  $('#imOut').textContent=`${w} × ${h} × ${c} = ${(w*h*c).toLocaleString()} 个数\n在 PyTorch 里：torch.zeros((${h}, ${w}, ${c})).shape → torch.Size([${h}, ${w}, ${c}])`}
['#imW','#imH','#imC'].forEach(s=>$(s).addEventListener('input',imRender));

// ---------- 第 1 天：过拟合 ----------
let ofSeed=3;const of={};
function lcg(s){return()=>{s=(s*1664525+1013904223)>>>0;return s/4294967296}}
function ofData(){const r=lcg(ofSeed*7919+11),g=()=>{let u=0;while(!u)u=r();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*r())};const f=x=>Math.sin(Math.PI*x);
  of.tr=Array.from({length:10},(_,i)=>{const x=-0.95+i*1.9/9+(r()-.5)*.08;return[x,f(x)+g()*.22]});of.te=Array.from({length:10},()=>{const x=r()*1.9-.95;return[x,f(x)+g()*.22]});of.errs=[];for(let d=0;d<=9;d++){const c=fit(d);of.errs.push([mse(c,of.tr),mse(c,of.te)])}}
function fit(d){const n=d+1,A=Array.from({length:n},()=>new Float64Array(n+1));for(const[x,y]of of.tr){const p=[];for(let i=0;i<n;i++)p.push(x**i);for(let i=0;i<n;i++){for(let j=0;j<n;j++)A[i][j]+=p[i]*p[j];A[i][n]+=p[i]*y}}
  for(let i=0;i<n;i++)A[i][i]+=1e-9;for(let i=0;i<n;i++){let m=i;for(let k=i+1;k<n;k++)if(Math.abs(A[k][i])>Math.abs(A[m][i]))m=k;[A[i],A[m]]=[A[m],A[i]];for(let k=0;k<n;k++){if(k===i)continue;const f=A[k][i]/A[i][i];for(let j=i;j<=n;j++)A[k][j]-=f*A[i][j]}}return A.map((r,i)=>r[n]/r[i])}
const poly=(c,x)=>c.reduce((s,a,i)=>s+a*x**i,0),mse=(c,P)=>P.reduce((s,[x,y])=>s+(poly(c,x)-y)**2,0)/P.length;
function ofRender(){const host=$('#ofHost');if(!host.offsetParent)return;const d=+$('#ofDeg').value;$('#ofDegv').textContent=d;const c=fit(d),C=colors();
  const W=Math.max(260,host.clientWidth),H=Math.round(Math.min(300,W*.68)),ctx=fitCanvas($('#ofCv'),W,H);const sx=x=>(x+1.05)/2.1*W,sy=y=>H/2-y*H/2/1.9;
  ctx.strokeStyle=C.line;ctx.beginPath();ctx.moveTo(0,sy(0));ctx.lineTo(W,sy(0));ctx.stroke();
  ctx.save();ctx.beginPath();ctx.rect(0,0,W,H);ctx.clip();ctx.strokeStyle=C.accent;ctx.lineWidth=2.2;ctx.beginPath();for(let i=0;i<=300;i++){const x=-1.05+2.1*i/300,y=Math.max(-5,Math.min(5,poly(c,x)));i?ctx.lineTo(sx(x),sy(y)):ctx.moveTo(sx(x),sy(y))}ctx.stroke();ctx.restore();
  for(const[x,y]of of.te){ctx.beginPath();ctx.arc(sx(x),sy(y),4.5,0,7);ctx.strokeStyle=C.neg;ctx.lineWidth=2;ctx.stroke()}
  for(const[x,y]of of.tr){ctx.beginPath();ctx.arc(sx(x),sy(y),4.5,0,7);ctx.fillStyle=C.pos;ctx.fill()}
  ctx.font=bodyFont(12);ctx.fillStyle=C.ink;ctx.fillText('● 训练点',8,16);ctx.fillText('○ 测试点',8,32);
  const cv2=$('#ofErr'),W2=Math.max(240,cv2.parentElement.clientWidth),H2=200,c2=fitCanvas(cv2,W2,H2),pl=44,pr=74,pt=10,pb=26;
  const E=of.errs.map(([a,b])=>[Math.log10(Math.max(a,1e-4)),Math.log10(Math.max(b,1e-4))]);let lo=Math.floor(Math.min(...E.flat())),hi=Math.ceil(Math.max(...E.flat()));if(hi===lo)hi++;
  const X=i=>pl+i/9*(W2-pl-pr),Y=v=>pt+(hi-v)/(hi-lo)*(H2-pt-pb);c2.font=MONO;c2.fillStyle=C.muted;c2.strokeStyle=C.line;c2.textAlign='right';c2.textBaseline='middle';
  for(let k=lo;k<=hi;k++){c2.beginPath();c2.moveTo(pl,Y(k));c2.lineTo(W2-pr,Y(k));c2.stroke();c2.fillText(k>=0?String(10**k):(10**k).toFixed(-k),pl-5,Y(k))}
  c2.textAlign='center';c2.textBaseline='top';for(let i=0;i<=9;i++)c2.fillText(i,X(i),H2-pb+5);
  c2.strokeStyle=C.ink;c2.globalAlpha=.25;c2.beginPath();c2.moveTo(X(d),pt);c2.lineTo(X(d),H2-pb);c2.stroke();c2.globalAlpha=1;
  const line=(j,col,name)=>{c2.strokeStyle=col;c2.lineWidth=2;c2.beginPath();E.forEach((e,i)=>i?c2.lineTo(X(i),Y(e[j])):c2.moveTo(X(i),Y(e[j])));c2.stroke();E.forEach((e,i)=>{c2.beginPath();c2.arc(X(i),Y(e[j]),i===d?5:3,0,7);c2.fillStyle=col;c2.fill()});c2.fillStyle=C.ink;c2.font=bodyFont(11.5);c2.textAlign='left';c2.textBaseline='middle';c2.fillText(name,W2-pr+8,Y(E[9][j])+(j?6:-6))};
  line(0,C.pos,'训练误差');line(1,C.neg,'测试误差');
  const[a,b]=of.errs[d];$('#ofCalc').textContent=`复杂度 ${d}（${d+1} 个参数）\n训练误差 ${a.toFixed(4)}\n测试误差 ${b.toFixed(4)}\n${d>=8?'训练误差很小、测试误差大：过拟合':d<=1?'两个误差都大：欠拟合':'两个误差都比较小：刚刚好'}`}
$('#ofDeg').oninput=ofRender;$('#ofNew').onclick=()=>{ofSeed++;ofData();ofRender()};

// ---------- 第 1 天：问题类型 & 毒蘑菇 & 猜形状 ----------
const KINDS=['回归','分类','标记','推荐','序列学习','无监督学习','强化学习'];
const KQ=[['预测一套房子能卖多少钱','回归','问的是“多少钱”，标签是一个数值。'],['识别手写数字是 0 到 9 中的哪一个','分类','问的是“哪一个”，有 10 个类别。'],['一张图里有猫、公鸡、狗和驴，把它们全都标出来','标记','一张图可以同时有好几个标签，书里叫标记问题（多标签分类）。'],
  ['根据你的购买记录，给你推荐可能喜欢的商品','推荐','书里专门讲了推荐系统，它要给每个用户排出一个列表。'],['把一句英语翻译成法语','序列学习','输入和输出都是长度不定的序列。'],['没有任何标签，让电脑把顾客自动分成几群','无监督学习','没有标签，让机器自己找结构，这是聚类。'],
  ['让程序下围棋，赢了得分、输了扣分，自己摸索怎么下','强化学习','和环境互动、靠奖励学习。'],['预测未来 6 小时的降雨量','回归','书里原例子：问的是“多少”。'],
  ['你的论文：看学生做题记录，猜他下一题会不会做对','分类','对或错两类，是二分类。因为输入是一串做题记录，它同时也是序列学习。'],['你的论文：根据学生情况决定下一题出什么','强化学习','这是做决策、看长期效果，论文里用的离线评估方法就来自强化学习。']];
function kqRender(){const ans=store.get('kq',{});$('#kQuiz').innerHTML=KQ.map(([q,a,why],i)=>{const pick=ans[i];return`<div class="q"><div>${i+1}. ${esc(q)}</div><div class="opts">${KINDS.map(k=>`<button class="btn sm${pick!=null&&k===a?' right':''}${pick===k&&k!==a?' wrong':''}" data-i="${i}" data-k="${k}">${k}</button>`).join('')}</div>${pick!=null?`<div class="fb">${pick===a?'<span class="ok">对。</span>':`<span class="bad">不对，是${a}。</span>`}${esc(why)}</div>`:''}</div>`}).join('');
  $$('#kQuiz button').forEach(b=>b.onclick=()=>{const a=store.get('kq',{});a[b.dataset.i]=b.dataset.k;store.set('kq',a);kqRender()});
  const got=Object.entries(ans).filter(([i,k])=>KQ[i]&&KQ[i][1]===k).length;$('#kScore').innerHTML=`答对 ${got} / ${KQ.length}　<button class="btn sm" id="kqReset">重做</button>`;$('#kqReset').onclick=()=>{store.set('kq',{});kqRender()}}
function muRender(){const p=+$('#muP').value,e=+$('#muC').value,C=e>=4?Infinity:Math.round(10**e);$('#muPv').textContent=p.toFixed(2);$('#muCv').textContent=C===Infinity?'∞（书里的设定）':C.toLocaleString();
  const eat=p===0?0:p*C,drop=(1-p)*1;const th=C===Infinity?0:1/(C+1);
  $('#muOut').textContent=`吃掉的预期损失 = ${p.toFixed(2)} × ${C===Infinity?'∞':C} + ${(1-p).toFixed(2)} × 0 = ${eat===Infinity?'∞':eat.toFixed(2)}\n扔掉的预期损失 = ${p.toFixed(2)} × 0 + ${(1-p).toFixed(2)} × 1 = ${drop.toFixed(2)}\n\n决定：${eat<drop?'吃':'扔掉'}\n\n${C===Infinity?'代价无穷大时，只要有一点点可能有毒就该扔。':`只有“有毒”概率低于 1/(${C}+1) ≈ ${th.toPrecision(2)} 时才值得吃。`}\n模型最可能的答案是“${p<0.5?'无毒':'有毒'}”，但该做的决定是“${eat<drop?'吃':'扔'}”。`}
['#muP','#muC'].forEach(s=>$(s).addEventListener('input',muRender));
const SQ=['torch.zeros((2, 3, 4))','torch.arange(12).reshape(-1, 6)','X = torch.arange(12).reshape(3, 4)\nX[1:3]','X = torch.arange(12).reshape(3, 4)\ntorch.cat((X, X), dim=1)','torch.arange(3).reshape(3, 1) + torch.arange(2).reshape(1, 2)','X = torch.arange(12).reshape(3, 4)\nX.sum(dim=0)','torch.zeros((256, 100, 64))[0, -1]'];
function sqTruth(code){const env=MT.newEnv();MT.runCell('import torch',env);const r=MT.runCell(code,env);return r.value&&MT.isT(r.value)?{shape:r.value.shape,text:r.text}:null}
function sqRender(){$('#shapeQuiz').innerHTML=SQ.map((code,i)=>`<div class="q"><pre class="calc" style="margin:0">${esc(code)}</pre><div class="row"><label class="small muted" for="sq${i}">结果的形状</label><input type="text" id="sq${i}" placeholder="例如 (3, 4)" size="14"><button class="btn sm primary" data-i="${i}">检查</button></div><div class="fb" id="sqf${i}"></div></div>`).join('');
  $$('#shapeQuiz button').forEach(b=>b.onclick=()=>{const i=+b.dataset.i,t=sqTruth(SQ[i]),guess=($('#sq'+i).value.match(/\d+/g)||[]).map(Number);const ok=guess.join()===t.shape.join();
    $('#sqf'+i).innerHTML=`${ok?'<span class="ok">对。</span>':`<span class="bad">实际是 ${shapeStr(t.shape)}。</span>`}运行结果：<pre class="calc" style="margin-top:6px">${esc(t.text)}</pre>`})}

// ---------- 第 2 天：点积和夹角 ----------
function dotRender(){const host=$('#dotHost');if(!host.offsetParent)return;const C=colors(),A=+$('#dotA').value,Bd=+$('#dotB').value;$('#dotAv').textContent=A+'°';$('#dotBv').textContent=Bd+'°';
  const W=Math.max(220,Math.min(320,host.clientWidth)),H=W,ctx=fitCanvas($('#dotCv'),W,H),cx=W/2,cy=H/2,R=W*0.38;
  ctx.strokeStyle=C.line;ctx.beginPath();ctx.arc(cx,cy,R,0,7);ctx.stroke();ctx.beginPath();ctx.moveTo(cx-R-10,cy);ctx.lineTo(cx+R+10,cy);ctx.moveTo(cx,cy-R-10);ctx.lineTo(cx,cy+R+10);ctx.stroke();
  const vec=(deg,col,lab)=>{const r=deg*Math.PI/180,x=cx+R*Math.cos(r),y=cy-R*Math.sin(r);ctx.strokeStyle=col;ctx.fillStyle=col;ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(cx,cy);ctx.lineTo(x,y);ctx.stroke();ctx.beginPath();ctx.arc(x,y,5,0,7);ctx.fill();ctx.font=bodyFont(13);ctx.fillStyle=C.ink;ctx.fillText(lab,x+(Math.cos(r)>0?8:-18),y-(Math.sin(r)>0?8:-14))};
  vec(A,C.accent,'a');vec(Bd,C.pos,'b');
  const ra=A*Math.PI/180,rb=Bd*Math.PI/180,a=[Math.cos(ra),Math.sin(ra)],b=[Math.cos(rb),Math.sin(rb)],dot=a[0]*b[0]+a[1]*b[1];let ang=Math.abs(A-Bd)%360;if(ang>180)ang=360-ang;
  $('#dotCalc').textContent=`a = (${a[0].toFixed(2)}, ${a[1].toFixed(2)})\nb = (${b[0].toFixed(2)}, ${b[1].toFixed(2)})\n点积 = ${a[0].toFixed(2)}×${b[0].toFixed(2)} + ${a[1].toFixed(2)}×${b[1].toFixed(2)}\n     = ${dot.toFixed(3)}\n夹角 ${ang.toFixed(0)}°，cos ${ang.toFixed(0)}° = ${Math.cos(ang*Math.PI/180).toFixed(3)}\n${dot>0.9?'方向几乎一样':dot>0.2?'方向比较像':dot>-0.2?'差不多垂直，互不相关':dot>-0.9?'方向比较相反':'方向几乎相反'}`}
['#dotA','#dotB'].forEach(s=>$(s).addEventListener('input',dotRender));
// ---------- 第 2 天：L1 和 L2 ----------
const nv={x:3,y:-4};
function normRender(){const host=$('#normHost');if(!host.offsetParent)return;const C=colors(),W=Math.max(220,Math.min(320,host.clientWidth)),H=W,ctx=fitCanvas($('#normCv'),W,H),u=W/12,cx=W/2,cy=H/2;
  ctx.strokeStyle=C.line;ctx.lineWidth=1;for(let i=-6;i<=6;i++){ctx.beginPath();ctx.moveTo(cx+i*u,0);ctx.lineTo(cx+i*u,H);ctx.stroke();ctx.beginPath();ctx.moveTo(0,cy+i*u);ctx.lineTo(W,cy+i*u);ctx.stroke()}
  ctx.strokeStyle=C.muted;ctx.beginPath();ctx.moveTo(0,cy);ctx.lineTo(W,cy);ctx.moveTo(cx,0);ctx.lineTo(cx,H);ctx.stroke();
  const px=cx+nv.x*u,py=cy-nv.y*u;ctx.strokeStyle=C.pos;ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(cx,cy);ctx.lineTo(px,cy);ctx.lineTo(px,py);ctx.stroke();
  ctx.strokeStyle=C.accent;ctx.beginPath();ctx.moveTo(cx,cy);ctx.lineTo(px,py);ctx.stroke();ctx.fillStyle=C.accent;ctx.beginPath();ctx.arc(px,py,5,0,7);ctx.fill();
  const l2=Math.hypot(nv.x,nv.y),l1=Math.abs(nv.x)+Math.abs(nv.y);
  $('#normCalc').textContent=`u = (${nv.x}, ${nv.y})\n\nL2 = √(${nv.x}² + ${nv.y}²) = √${nv.x*nv.x+nv.y*nv.y} = ${l2.toFixed(3)}\nL1 = |${nv.x}| + |${nv.y}| = ${l1}\n\ntorch.norm(u)       → tensor(${l2.toFixed(4)})\ntorch.abs(u).sum()  → tensor(${l1}.)\n\nL1 总是 ≥ L2，只有向量落在坐标轴上时两者相等。`}
$('#normCv').addEventListener('click',e=>{const cv=$('#normCv'),r=cv.getBoundingClientRect(),u=r.width/12;nv.x=Math.round((e.clientX-r.left-r.width/2)/u);nv.y=Math.round(-(e.clientY-r.top-r.height/2)/u);normRender()});

// ---------- 第 3 天：割线和切线 ----------
const fq=x=>3*x*x-4*x;
function secRender(){const host=$('#secHost');if(!host.offsetParent)return;const C=colors(),x0=+$('#secX').value,h=10**+$('#secH').value;$('#secXv').textContent=x0.toFixed(1);$('#secHv').textContent=h<0.01?h.toExponential(1):h.toFixed(3);
  const W=Math.max(260,host.clientWidth),H=Math.round(Math.min(300,W*.7)),ctx=fitCanvas($('#secCv'),W,H),X0=-0.5,X1=3.5,Y0=-3,Y1=18;const sx=x=>(x-X0)/(X1-X0)*W,sy=y=>H-(y-Y0)/(Y1-Y0)*H;
  ctx.strokeStyle=C.line;ctx.beginPath();ctx.moveTo(0,sy(0));ctx.lineTo(W,sy(0));ctx.moveTo(sx(0),0);ctx.lineTo(sx(0),H);ctx.stroke();
  ctx.strokeStyle=C.accent;ctx.lineWidth=2.4;ctx.beginPath();for(let i=0;i<=200;i++){const x=X0+(X1-X0)*i/200;i?ctx.lineTo(sx(x),sy(fq(x))):ctx.moveTo(sx(x),sy(fq(x)))}ctx.stroke();
  const slope=6*x0-4,sec=(fq(x0+h)-fq(x0))/h;const line=(m,col,dash)=>{ctx.strokeStyle=col;ctx.lineWidth=2;ctx.setLineDash(dash);ctx.beginPath();ctx.moveTo(sx(X0),sy(fq(x0)+m*(X0-x0)));ctx.lineTo(sx(X1),sy(fq(x0)+m*(X1-x0)));ctx.stroke();ctx.setLineDash([])};
  line(slope,C.ink,[5,4]);line(sec,C.neg,[]);
  for(const[x,col]of[[x0,C.pos],[x0+h,C.neg]]){ctx.fillStyle=col;ctx.beginPath();ctx.arc(sx(x),sy(fq(x)),5,0,7);ctx.fill()}
  $('#secCalc').textContent=`f(x) = 3x² − 4x\n割线斜率 = (f(${x0.toFixed(1)} + h) − f(${x0.toFixed(1)})) / h\n         = ${sec.toFixed(5)}\n切线斜率 f'(${x0.toFixed(1)}) = 6×${x0.toFixed(1)} − 4 = ${slope.toFixed(5)}\n两者相差 ${(sec-slope).toFixed(5)}，正好是 3h`}
['#secX','#secH'].forEach(s=>$(s).addEventListener('input',secRender));
// ---------- 第 3 天：梯度下降 ----------
const gd={p:[2.6,1.6],path:[[2.6,1.6]]};const gf=(a,b)=>a*a+3*b*b,gg=(a,b)=>[2*a,6*b];
function gradRender(){const host=$('#gradHost');if(!host.offsetParent)return;const C=colors(),cs=getComputedStyle(document.documentElement);const W=Math.max(240,Math.min(360,host.clientWidth)),H=W,ctx=fitCanvas($('#gradCv'),W,H),L=3.2;
  const sx=x=>(x+L)/(2*L)*W,sy=y=>(L-y)/(2*L)*H;const N=48,cw=W/N;
  for(let i=0;i<N;i++)for(let j=0;j<N;j++){const x=-L+(i+.5)*2*L/N,y=L-(j+.5)*2*L/N,v=Math.min(1,gf(x,y)/30);ctx.fillStyle=`color-mix(in srgb, ${cs.getPropertyValue('--accent').trim()} ${Math.round(v*70)}%, ${C.surface})`;ctx.fillRect(i*cw,j*cw,cw+1,cw+1)}
  ctx.strokeStyle=C.surface;ctx.globalAlpha=.6;for(const lv of[1,3,6,10,15,21,28]){ctx.beginPath();for(let k=0;k<=80;k++){const t=k/80*2*Math.PI,x=Math.sqrt(lv)*Math.cos(t),y=Math.sqrt(lv/3)*Math.sin(t);k?ctx.lineTo(sx(x),sy(y)):ctx.moveTo(sx(x),sy(y))}ctx.stroke()}ctx.globalAlpha=1;
  ctx.strokeStyle=C.pos;ctx.lineWidth=2;ctx.beginPath();gd.path.forEach(([a,b],i)=>i?ctx.lineTo(sx(a),sy(b)):ctx.moveTo(sx(a),sy(b)));ctx.stroke();
  const[a,b]=gd.p,[ga,gb]=gg(a,b);const n=Math.hypot(ga,gb)||1,len=Math.min(60,20+n*4);ctx.strokeStyle=C.ink;ctx.fillStyle=C.ink;ctx.lineWidth=2.5;const ex=sx(a)-ga/n*len,ey=sy(b)+gb/n*len;ctx.beginPath();ctx.moveTo(sx(a),sy(b));ctx.lineTo(ex,ey);ctx.stroke();
  const ang=Math.atan2(ey-sy(b),ex-sx(a));ctx.beginPath();ctx.moveTo(ex,ey);ctx.lineTo(ex-9*Math.cos(ang-0.4),ey-9*Math.sin(ang-0.4));ctx.lineTo(ex-9*Math.cos(ang+0.4),ey-9*Math.sin(ang+0.4));ctx.closePath();ctx.fill();
  ctx.fillStyle=C.pos;ctx.beginPath();ctx.arc(sx(a),sy(b),6,0,7);ctx.fill();ctx.strokeStyle=C.surface;ctx.lineWidth=2;ctx.stroke();
  const lr=+$('#gradLr').value;$('#gradLrv').textContent=lr.toFixed(2);
  $('#gradCalc').textContent=`第 ${gd.path.length-1} 步\n位置 (x₁, x₂) = (${a.toFixed(3)}, ${b.toFixed(3)})\nf = x₁² + 3x₂² = ${gf(a,b).toFixed(4)}\n梯度 ∇f = (2x₁, 6x₂) = (${ga.toFixed(3)}, ${gb.toFixed(3)})\n下一步 = 位置 − ${lr} × 梯度\n       = (${(a-lr*ga).toFixed(3)}, ${(b-lr*gb).toFixed(3)})${Math.abs(b)>50?'\n\n发散了：学习率太大，降低后点“回到起点”。':''}`}
function gradStep(n){const lr=+$('#gradLr').value;for(let i=0;i<n;i++){const[a,b]=gd.p,[ga,gb]=gg(a,b);gd.p=[a-lr*ga,b-lr*gb];if(Math.abs(gd.p[1])>1e6)break;gd.path.push(gd.p)}gradRender()}
$('#gradStep').onclick=()=>gradStep(1);$('#gradRun').onclick=()=>gradStep(10);$('#gradReset').onclick=()=>{gd.p=gd.path[0];gd.path=[gd.p];gradRender()};$('#gradLr').oninput=gradRender;
$('#gradCv').addEventListener('click',e=>{const cv=$('#gradCv'),r=cv.getBoundingClientRect(),L=3.2;const x=(e.clientX-r.left)/r.width*2*L-L,y=L-(e.clientY-r.top)/r.height*2*L;gd.p=[x,y];gd.path=[gd.p];gradRender()});


// ---------- 第 4 天：灌铅骰子 ----------
const dice={counts:[0,0,0,0,0,0],n:0};
function diceProbs(){const w=+$('#diceW').value,o=(1-w)/5;return[o,o,o,o,o,w]}
function diceRoll(k){const p=diceProbs();for(let t=0;t<k;t++){let u=Math.random(),i=0;for(;i<5;i++){u-=p[i];if(u<=0)break}dice.counts[i]++;dice.n++}diceRender()}
function diceRender(){const host=$('#diceHost');if(!host.offsetParent)return;const C=colors(),p=diceProbs();$('#diceWv').textContent=p[5].toFixed(3);
  const W=Math.max(240,Math.min(420,host.clientWidth)),H=220,ctx=fitCanvas($('#diceCv'),W,H),pl=36,pb=24,pt=10,bw=(W-pl-10)/6;const top=Math.max(0.4,...p.map(x=>x*1.3),...dice.counts.map(c=>dice.n?c/dice.n*1.1:0));const sy=v=>pt+(1-v/top)*(H-pt-pb);
  ctx.font=MONO;ctx.fillStyle=C.muted;ctx.strokeStyle=C.line;ctx.textAlign='right';ctx.textBaseline='middle';for(const t of[0,0.1,0.2,0.3,0.4,0.5,0.6].filter(t=>t<=top)){ctx.beginPath();ctx.moveTo(pl,sy(t));ctx.lineTo(W-10,sy(t));ctx.stroke();ctx.fillText(t.toFixed(1),pl-5,sy(t))}
  for(let i=0;i<6;i++){const x=pl+i*bw+bw*.18,w=bw*.64,f=dice.n?dice.counts[i]/dice.n:0;ctx.fillStyle=i===5?C.pos:C.accent;ctx.beginPath();ctx.roundRect?ctx.roundRect(x,sy(f),w,sy(0)-sy(f),[4,4,0,0]):ctx.rect(x,sy(f),w,sy(0)-sy(f));ctx.fill();
    ctx.strokeStyle=C.ink;ctx.setLineDash([4,3]);ctx.beginPath();ctx.moveTo(x-4,sy(p[i]));ctx.lineTo(x+w+4,sy(p[i]));ctx.stroke();ctx.setLineDash([]);ctx.fillStyle=C.muted;ctx.textAlign='center';ctx.textBaseline='top';ctx.fillText(String(i+1),x+w/2,H-pb+6)}
  const est=dice.counts.map(c=>dice.n?c/dice.n:0);
  $('#diceCalc').textContent=`已经掷了 ${dice.n} 次\n\n点数  次数   估计概率   真实概率\n`+est.map((e,i)=>`  ${i+1}   ${String(dice.counts[i]).padStart(5)}    ${e.toFixed(3)}      ${p[i].toFixed(3)}`).join('\n')+`\n\n${dice.n<100?'次数太少，估计值波动很大。':dice.n<2000?'估计值开始靠近真实概率了。':'掷得够多了，估计值和真实概率很接近。'}`}
$$('[data-roll]').forEach(b=>b.onclick=()=>diceRoll(+b.dataset.roll));$('#diceReset').onclick=()=>{dice.counts=[0,0,0,0,0,0];dice.n=0;diceRender()};$('#diceW').oninput=()=>{dice.counts=[0,0,0,0,0,0];dice.n=0;diceRender()};

// ---------- 第 4 天：一万个人的贝叶斯 ----------
function bayesRender(){const prev=10**+$('#byP').value,sens=+$('#byS').value,fpr=+$('#byF').value;$('#byPv').textContent=(prev*100).toPrecision(2)+'%';$('#bySv').textContent=Math.round(sens*100)+'%';$('#byFv').textContent=(fpr*100).toFixed(1)+'%';
  const N=10000,sick=N*prev,healthy=N-sick,tp=sick*sens,fp=healthy*fpr,post=tp/(tp+fp||1);
  $('#byBar').innerHTML=`<div class="popbar"><i class="tp" style="width:${(post*100).toFixed(2)}%"></i><i class="fp" style="flex:1"></i></div><div class="legend"><span><i style="background:var(--pos)"></i>真的感染：${tp.toFixed(1)} 人</span><span><i style="background:var(--muted)"></i>误报的健康人：${fp.toFixed(1)} 人</span></div>`;
  $('#byCalc').textContent=`一万个人里：感染 ${sick.toFixed(1)} 人，健康 ${healthy.toFixed(1)} 人\n感染者里查出阳性：${sick.toFixed(1)} × ${sens.toFixed(2)} = ${tp.toFixed(1)} 人\n健康人里误报阳性：${healthy.toFixed(1)} × ${fpr.toFixed(3)} = ${fp.toFixed(1)} 人\n\nP(感染 | 阳性) = ${tp.toFixed(1)} ÷ (${tp.toFixed(1)} + ${fp.toFixed(1)}) = ${(post*100).toFixed(2)}%\n\n${prev<0.01?'病越罕见，阳性结果里误报的比例越大。':'病一旦常见起来，阳性结果就可信多了。'}`}
['#byP','#byS','#byF'].forEach(s=>$(s).addEventListener('input',bayesRender));

// ---------- 第 4 天：BKT 更新 ----------
const bk={obs:[]};
function bkParams(){const v=k=>+$(k).value;return{p0:v('#bkP0'),g:v('#bkG'),s:v('#bkS'),l:v('#bkL')}}
function bkTrace(){const{p0,g,s,l}=bkParams();const ps=[p0],steps=[];let p=p0;for(const y of bk.obs){const pc=p*(1-s)+(1-p)*g;const post=y?p*(1-s)/pc:p*s/(1-pc);const nx=post+(1-post)*l;steps.push({p,pc,post,nx,y});p=nx;ps.push(p)}return{ps,steps}}
function bktRender(){const host=$('#bktHost');if(!host.offsetParent)return;const C=colors(),P=bkParams();['P0','G','S','L'].forEach(k=>$('#bk'+k+'v').textContent=(+$('#bk'+k).value).toFixed(2));
  const{ps,steps}=bkTrace();const W=Math.max(240,Math.min(460,host.clientWidth)),H=200,ctx=fitCanvas($('#bktCv'),W,H),pl=36,pr=12,pt=10,pb=24;const n=Math.max(8,ps.length-1);const sx=i=>pl+i/n*(W-pl-pr),sy=v=>pt+(1-v)*(H-pt-pb);
  ctx.font=MONO;ctx.fillStyle=C.muted;ctx.strokeStyle=C.line;ctx.textAlign='right';ctx.textBaseline='middle';for(const t of[0,0.25,0.5,0.75,1]){ctx.beginPath();ctx.moveTo(pl,sy(t));ctx.lineTo(W-pr,sy(t));ctx.stroke();ctx.fillText(t.toFixed(2),pl-5,sy(t))}
  ctx.textAlign='center';ctx.textBaseline='top';for(let i=0;i<=n;i+=Math.max(1,Math.ceil(n/8)))ctx.fillText(i,sx(i),H-pb+6);
  ctx.strokeStyle=C.accent;ctx.lineWidth=2;ctx.beginPath();ps.forEach((v,i)=>i?ctx.lineTo(sx(i),sy(v)):ctx.moveTo(sx(i),sy(v)));ctx.stroke();
  ps.forEach((v,i)=>{ctx.beginPath();ctx.arc(sx(i),sy(v),4.5,0,7);ctx.fillStyle=i===0?C.muted:(bk.obs[i-1]?C.neg:C.pos);ctx.fill();ctx.strokeStyle=C.surface;ctx.lineWidth=2;ctx.stroke()});
  $('#bkSeq').textContent=bk.obs.length?'记录：'+bk.obs.map(y=>y?'对':'错').join(' '):'还没有答题记录';
  const last=steps[steps.length-1];
  $('#bktCalc').textContent=last?`第 ${steps.length} 题${last.y?'答对':'答错'}\n答题前 P(会) = ${last.p.toFixed(3)}\n预测这题答对的概率 = ${last.p.toFixed(3)}×${(1-P.s).toFixed(2)} + ${(1-last.p).toFixed(3)}×${P.g.toFixed(2)} = ${last.pc.toFixed(3)}\n\n看到${last.y?'答对':'答错'}后（贝叶斯定理）：\nP(会 | ${last.y?'对':'错'}) = ${last.post.toFixed(3)}\n再加上这次可能学会（learn = ${P.l.toFixed(2)}）：\n${last.post.toFixed(3)} + (1 − ${last.post.toFixed(3)}) × ${P.l.toFixed(2)} = ${last.nx.toFixed(3)}\n\n蓝点 = 答对后，橙点 = 答错后`:`现在 P(会) = p0 = ${P.p0.toFixed(2)}\n模型预测下一题答对的概率 = ${(P.p0*(1-P.s)+(1-P.p0)*P.g).toFixed(3)}\n点“答对”或“答错”开始。`}
$('#bkYes').onclick=()=>{bk.obs.push(1);bktRender()};$('#bkNo').onclick=()=>{bk.obs.push(0);bktRender()};$('#bkReset').onclick=()=>{bk.obs=[];bktRender()};['#bkP0','#bkG','#bkS','#bkL'].forEach(s=>$(s).addEventListener('input',bktRender));

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

// ---------- 第 6 天：softmax 调试台 ----------
const SMN=['猫','鸡','狗'];
function smRender(){const o=[1,2,3].map(i=>+$('#smO'+i).value),T=+$('#smT').value,y=+$('#smY').value;[1,2,3].forEach(i=>$('#smO'+i+'v').textContent=(+$('#smO'+i).value).toFixed(1));$('#smTv').textContent=T.toFixed(1);
  const z=o.map(v=>v/T),m=Math.max(...z),e=z.map(v=>Math.exp(v-m)),s=e.reduce((a,b)=>a+b,0),p=e.map(v=>v/s);const best=p.indexOf(Math.max(...p));
  $('#smBars').innerHTML=p.map((v,i)=>`<div class="bar"><span class="ch">${SMN[i]}</span><span class="tr"><i style="width:${(v*100).toFixed(1)}%;${i===y?'background:var(--pos)':''}"></i></span><span class="pv">${(v*100).toFixed(1)}%${i===best?' ◀':''}</span></div>`).join('');
  const loss=-Math.log(p[y]),H=-p.reduce((a,v)=>a+(v>0?v*Math.log2(v):0),0);
  $('#smCalc').textContent=`logit / 温度 = [${z.map(v=>v.toFixed(2)).join(', ')}]\nexp 之后    = [${z.map(v=>Math.exp(v).toFixed(2)).join(', ')}]，加起来 ${z.reduce((a,v)=>a+Math.exp(v),0).toFixed(2)}\nsoftmax     = [${p.map(v=>v.toFixed(3)).join(', ')}]，加起来 1\n\n预测：${SMN[best]}（logit 最大的那个）${best===y?'，猜对了':'，猜错了'}\n交叉熵损失 = −log ŷ_${SMN[y]} = −log ${p[y].toFixed(3)} = ${loss.toFixed(3)}\n对 logit 的梯度 = ŷ − y = [${p.map((v,i)=>(v-(i===y?1:0)).toFixed(3)).join(', ')}]\n  （正确类别的梯度是负的，梯度下降会把它的 logit 往上推）\n预测分布的熵 = ${H.toFixed(3)} 比特（最大 ${Math.log2(3).toFixed(3)}，表示完全拿不准）`}
['#smO1','#smO2','#smO3','#smT'].forEach(s=>$(s).addEventListener('input',smRender));$('#smY').onchange=smRender;

// ---------- 第 6 天：可靠性图和 ECE ----------
let calSeed=5;
function calData(){let s=calSeed*9973+1;const r=()=>{s=(s*1664525+1013904223)>>>0;return s/4294967296};const n=3000,ps=[],ys=[];for(let i=0;i<n;i++){const p=0.05+0.9*r();ps.push(p);ys.push(r()<p?1:0)}return{ps,ys}}
let calD=null;
function calRender(){const host=$('#calHost');if(!host.offsetParent)return;if(!calD)calD=calData();const C=colors(),k=+$('#calK').value;$('#calKv').textContent=k.toFixed(1)+(Math.abs(k-1)<0.05?'（校准良好）':k>1?'（过度自信）':'（不够自信）');
  const lg=p=>Math.log(p/(1-p)),sg=x=>1/(1+Math.exp(-x));const conf=calD.ps.map(p=>sg(lg(p)*k));const B=10,cnt=new Array(B).fill(0),sc=new Array(B).fill(0),sy=new Array(B).fill(0);
  conf.forEach((c,i)=>{const b=Math.min(B-1,Math.floor(c*B));cnt[b]++;sc[b]+=c;sy[b]+=calD.ys[i]});const N=conf.length;let ece=0;for(let b=0;b<B;b++)if(cnt[b])ece+=cnt[b]/N*Math.abs(sy[b]/cnt[b]-sc[b]/cnt[b]);
  const W=Math.max(240,Math.min(380,host.clientWidth)),H=W,ctx=fitCanvas($('#calCv'),W,H),pl=40,pr=10,pt=10,pb=34;const X=v=>pl+v*(W-pl-pr),Y=v=>pt+(1-v)*(H-pt-pb);
  ctx.font=MONO;ctx.fillStyle=C.muted;ctx.strokeStyle=C.line;ctx.textAlign='right';ctx.textBaseline='middle';for(let t=0;t<=1.0001;t+=0.2){ctx.beginPath();ctx.moveTo(pl,Y(t));ctx.lineTo(W-pr,Y(t));ctx.stroke();ctx.fillText(t.toFixed(1),pl-5,Y(t))}
  ctx.textAlign='center';ctx.textBaseline='top';for(let t=0;t<=1.0001;t+=0.2)ctx.fillText(t.toFixed(1),X(t),H-pb+5);ctx.font=bodyFont(11.5);ctx.fillText('模型给出的置信度',(pl+W-pr)/2,H-16);
  for(let b=0;b<B;b++){if(!cnt[b])continue;const acc=sy[b]/cnt[b],x0=X(b/B)+2,x1=X((b+1)/B)-2;ctx.fillStyle=C.accent;ctx.fillRect(x0,Y(acc),x1-x0,Y(0)-Y(acc));const mc=sc[b]/cnt[b];ctx.fillStyle=C.pos;ctx.fillRect(x0,Y(mc)-1.5,x1-x0,3)}
  ctx.strokeStyle=C.ink;ctx.setLineDash([5,4]);ctx.beginPath();ctx.moveTo(X(0),Y(0));ctx.lineTo(X(1),Y(1));ctx.stroke();ctx.setLineDash([]);
  $('#calCalc').textContent=`${N} 个样本，分成 ${B} 个等宽区间\n\n区间        样本数  平均置信度  实际正确率\n`+cnt.map((c,b)=>c?`${(b/B).toFixed(1)}–${((b+1)/B).toFixed(1)}  ${String(c).padStart(6)}     ${(sc[b]/c).toFixed(3)}      ${(sy[b]/c).toFixed(3)}`:null).filter(Boolean).join('\n')+`\n\nECE = Σ (区间样本数 / 总数) × |实际正确率 − 平均置信度|\n    = ${(ece*100).toFixed(2)}%\n\n蓝柱 = 实际正确率，橙线 = 平均置信度，虚线 = 完美校准`}
$('#calK').oninput=calRender;$('#calNew').onclick=()=>{calSeed++;calD=calData();calRender()};

// ---------- 第 7 天：matplotlib 图（Animator 曲线、show_images 网格） ----------
const FMT_COLOR={m:'--purple',g:'--ok',r:'--bad',b:'--accent',k:'--ink',c:'--neg'};
function fmtStyle(fmt,i){const m=fmt.match(/^([bgrcmyk])?(--|-\.|-|:|o)?$/)||[];const ls=m[2]||'-';return{color:m[1]?FMT_COLOR[m[1]]:SERIESC[i%4],dash:ls==='--'?[6,4]:ls==='-.'?[7,3,2,3]:ls===':'?[2,3]:[],scatter:ls==='o'}}
const VIRIDIS=[[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]];
function viridis(t){t=Math.max(0,Math.min(1,t))*4;const i=Math.min(3,Math.floor(t)),f=t-i,a=VIRIDIS[i],b=VIRIDIS[i+1];return[a[0]+(b[0]-a[0])*f,a[1]+(b[1]-a[1])*f,a[2]+(b[2]-a[2])*f]}
function drawImage(cv,img,px){const{H,W}=img;cv.width=W;cv.height=H;cv.style.width=px+'px';cv.style.height=Math.round(px*H/W)+'px';const c=cv.getContext('2d'),d=c.createImageData(W,H);let lo=Infinity,hi=-Infinity;for(const v of img.px){if(v<lo)lo=v;if(v>hi)hi=v}const rg=hi-lo||1;
  const blues=t=>[Math.round(247-t*(247-8)),Math.round(251-t*(251-48)),Math.round(255-t*(255-107))],reds=t=>[Math.round(255-t*(255-103)),Math.round(245-t*245),Math.round(240-t*(240-13))];for(let k=0;k<W*H;k++){const[r,g,b]=img.cmap==='Reds'?reds((img.px[k]-lo)/rg):img.cmap==='Blues'?blues((img.px[k]-lo)/rg):viridis((img.px[k]-lo)/rg);d.data[k*4]=r;d.data[k*4+1]=g;d.data[k*4+2]=b;d.data[k*4+3]=255}c.putImageData(d,0,0)}
function renderFig(wrap,fig){if(fig.axes.some(a=>a.image)){const grid=document.createElement('div');grid.className='imgrid';grid.style.gridTemplateColumns=`repeat(${fig.cols},minmax(0,1fr))`;
    const hm=fig.axes.some(a=>a.image&&(a.image.cmap==='Reds'||a.image.cmap==='Blues')),sz=hm?Math.max(110,Math.min(260,Math.floor(((wrap.parentElement&&wrap.parentElement.clientWidth)||320)/fig.cols)-16)):72;
    for(const ax of fig.axes){const cell=document.createElement('figure');if(ax.title){const t=document.createElement('figcaption');t.textContent=ax.title;cell.appendChild(t)}if(ax.image){const cv=document.createElement('canvas');cv.setAttribute('aria-label',ax.title||(hm?'注意力权重热图':'图片'));cv.className=hm?'hm':'';cell.appendChild(cv);drawImage(cv,ax.image,sz)}
      if(hm&&(ax.xlabel||ax.ylabel)){const c=document.createElement('figcaption');c.className='small muted';c.textContent=`横轴（列）：${ax.xlabel||''}　纵轴（行）：${ax.ylabel||''}`;cell.appendChild(c)}grid.appendChild(cell)}
    if(hm){const n=document.createElement('p');n.className='small muted';n.textContent='颜色越深数值越大：最浅是这张图里的最小值，最深是最大值。';wrap.appendChild(grid);wrap.appendChild(n);return}wrap.appendChild(grid);return}
  for(const ax of fig.axes){if(!ax.lines.length)continue;const spec={series:ax.lines.map((l,i)=>({x:l.x,y:l.y,label:ax.legend[i]||'',...fmtStyle(l.fmt,i)})),hlines:[],xlabel:ax.xlabel,ylabel:ax.ylabel,legend:ax.legend.length>0,xlim:ax.xlim,ylim:ax.ylim,xscale:ax.xscale,yscale:ax.yscale};const cv=document.createElement('canvas');cv.setAttribute('aria-label','训练曲线');wrap.appendChild(cv);drawPlot(cv,spec)}}

// ---------- 第 7 天：解压内嵌的 Fashion-MNIST，像素查看器 ----------
let FM_U8=null;
const fmReady=(async()=>{const el=document.getElementById('fmData');const st=()=>document.getElementById('fmStatus');try{const bin=Uint8Array.from(atob(el.textContent.trim()),c=>c.charCodeAt(0));
  if(typeof DecompressionStream==='undefined')throw new Error('这个浏览器不支持解压');const buf=await new Response(new Blob([bin]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();FM_U8=new Uint8Array(buf);MT.setFashion(FM_U8);
  if(st())st().textContent='图片数据已就绪：训练集 6000 张，测试集 1000 张，来自 Fashion-MNIST 官方数据。';return true}catch(e){if(st()){st().textContent=`图片数据解压失败（${e.message}），请换用最新版的 Chrome、Edge、Safari 或 Firefox。`;st().classList.add('bad')}return false}})();
const FM_NAMES=['t-shirt','trouser','pullover','dress','coat','sandal','shirt','sneaker','bag','ankle boot'];
function pixRender(){if(!FM_U8)return;const i=+$('#pixI').value;$('#pixIv').textContent=i;const px=FM_U8.subarray(i*784,i*784+784),lab=FM_U8[6000*784+i];$('#pixLab2').textContent=`标签 ${lab}：${FM_NAMES[lab]}`;
  const cv=$('#pixCv'),c=cv.getContext('2d');for(let r=0;r<28;r++)for(let q=0;q<28;q++){const v=px[r*28+q];c.fillStyle=`rgb(${v},${v},${v})`;c.fillRect(q*10,r*10,10,10)}
  const s=$('#pixStrip'),W=Math.max(200,s.parentElement.clientWidth);s.width=784;s.style.width=W+'px';s.style.height='40px';const sc=s.getContext('2d');for(let k=0;k<784;k++){const v=px[k];sc.fillStyle=`rgb(${v},${v},${v})`;sc.fillRect(k,0,1,40)}}
$('#pixI').addEventListener('input',pixRender);
$('#pixCv').addEventListener('mousemove',e=>{if(!FM_U8)return;const r=$('#pixCv').getBoundingClientRect(),q=Math.floor((e.clientX-r.left)/r.width*28),row=Math.floor((e.clientY-r.top)/r.height*28);if(q<0||q>27||row<0||row>27)return;const i=+$('#pixI').value,v=FM_U8[i*784+row*28+q];
  $('#pixCalc').textContent=`第 ${row} 行第 ${q} 列
原始值 ${v}（0 到 255）
ToTensor 之后 = ${v} / 255 = ${(v/255).toFixed(4)}
张量里的位置：img[0, ${row}, ${q}]
拉平后的位置：第 ${row*28+q} 个（= ${row} × 28 + ${q}）`});
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
// ---------- 第 11 天：DKT 的参数清单 ----------
function pcRender(){const K=+$('#pcDs').value,d=+$('#pcD').value,g=$('#pcG').value==='1';$('#pcDv').textContent=d;
  const rows=[['emb.weight',[2*K,d],'嵌入表：每个（知识点，答对或答错）组合一行，共 2K 行'],['lstm.weight_ih_l0',[4*d,d],'LSTM 输入到 4 个门的权重（第 14 天讲 4 个门）'],['lstm.weight_hh_l0',[4*d,d],'LSTM 上一步隐状态到 4 个门的权重'],['lstm.bias_ih_l0',[4*d],'LSTM 偏置'],['lstm.bias_hh_l0',[4*d],'LSTM 偏置（PyTorch 有两份，加在一起用）'],['out.weight',[K,d],'输出头：隐状态到每个知识点的答对概率（logit）'],['out.bias',[K],'输出头偏置']];
  if(g)rows.push(['g.weight',[K,d],'策略头：给每个知识点的出题打分加一个修正，初始化为 0'],['g.bias',[K],'策略头偏置，初始化为 0']);
  const cnt=s=>s.reduce((a,b)=>a*b,1),tot=rows.reduce((a,r)=>a+cnt(r[1]),0),fmt=n=>n.toLocaleString('en-US');
  $('#pcTbl').innerHTML='<thead><tr><th>名字</th><th>形状</th><th class="num">个数</th><th>是什么</th></tr></thead><tbody>'+rows.map(([n,s,w])=>`<tr><td><code>${n}</code></td><td style="white-space:nowrap"><code>(${s.join(', ')})</code></td><td class="num">${fmt(cnt(s))}</td><td class="small">${w}</td></tr>`).join('')+`<tr><td><b>合计</b></td><td></td><td class="num"><b>${fmt(tot)}</b></td><td></td></tr></tbody>`;
  const lstm=2*4*d*d+2*4*d;$('#pcCalc').textContent=`一共 ${fmt(tot)} 个参数。\nLSTM 占 ${fmt(lstm)} 个（${(100*lstm/tot).toFixed(0)}%），和 K 无关，只看 d：2 × 4d² + 2 × 4d。\n和 K 有关的部分：嵌入 2Kd，输出头 Kd + K${g?'，策略头 Kd + K':''}。\n${K===681?'Junyi 有 681 个知识点，所以和 K 有关的部分比 LSTM 还大。':''}${d!==64?`\n你论文用的是 d = 64（D_MODEL = 64）。`:''}\n\n对照：真 PyTorch 里 sum(p.numel() for p in DKT(145, policy_head=True).parameters()) = 70,690。`}
['#pcDs','#pcG'].forEach(s=>$(s).onchange=pcRender);$('#pcD').addEventListener('input',pcRender);
// ---------- 第 12 天：做题记录变成词元 ----------
const TK={K:5,seq:[]};
function tkNew(){TK.seq=Array.from({length:10},()=>[Math.floor(Math.random()*TK.K),Math.random()<0.6?1:0]);tkRender()}
function tkRender(){const K=TK.K;if(!TK.seq.length){TK.seq=[[0,1],[0,1],[2,0],[2,0],[2,1],[1,1],[3,0],[3,1],[4,0],[0,1]]}
  $('#tkRow').innerHTML=TK.seq.map(([s,r],t)=>`<button class="tk ${r?'ok':'no'}" data-t="${t}" aria-label="第 ${t+1} 题，知识点 ${s}，${r?'答对':'答错'}"><span class="small muted">第 ${t+1} 题</span><span>知识点 ${s}</span><b>${r?'对':'错'}</b><code>${s+K*r}</code></button>`).join('');
  $$('#tkRow .tk').forEach(b=>b.onclick=()=>{const t=+b.dataset.t;TK.seq[t][1]^=1;tkRender()});
  const sk=TK.seq.map(x=>x[0]),rs=TK.seq.map(x=>x[1]),ix=TK.seq.map(([s,r])=>s+K*r);
  $('#tkCalc').textContent=`sk（知识点）  = [${sk.join(', ')}]\nrs（对错）    = [${rs.join(', ')}]\nsk + K * rs   = [${ix.join(', ')}]    K = ${K}\n\n编号 0 到 ${K-1}：知识点 0 到 ${K-1} 答错\n编号 ${K} 到 ${2*K-1}：知识点 0 到 ${K-1} 答对\n\nself.emb = nn.Embedding(2 * K, d) 是一张 ${2*K} 行、d 列的表，\n每个编号取出其中一行（d 维向量）作为这一步 LSTM 的输入。\n一个批量的形状：(批量, 时间步) 的整数 → 嵌入后 (批量, 时间步, d)。`}
$('#tkRand').onclick=tkNew;
// ---------- 第 12 天：随机抽样和顺序分区 ----------
function ptRender(){const mode=$('#ptMode').value,N=35,bs=2,ns=5,cell=new Array(N).fill(null),batches=[];
  if(mode==='seq'){const off=Math.floor(Math.random()*(ns+1)),nt=Math.floor((N-off-1)/bs)*bs,cols=nt/bs,nb=Math.floor(cols/ns);
    for(let b=0;b<nb;b++){const rows=[];for(let r=0;r<bs;r++){const st=off+r*cols+b*ns;rows.push(st);for(let k=0;k<ns;k++)cell[st+k]={b,r}}batches.push(rows)}
    ptDraw(cell,batches,ns,`顺序分区：随机偏移 offset = ${off}\n先取 ${nt} 个数排成 ${bs} 行，每行 ${cols} 个，再每 ${ns} 列切一个小批量，共 ${nb} 个。\n${batches.map((rows,b)=>`小批量 ${b+1}：X 的两行从 ${rows.join(' 和 ')} 开始`).join('\n')}\n\n同一行里，小批量 1 后面紧接着小批量 2：上一批最后的隐状态可以直接接着用。`)}
  else{const off=Math.floor(Math.random()*ns),ns2=Math.floor((N-off-1)/ns),starts=Array.from({length:ns2},(_,i)=>off+i*ns);for(let i=starts.length-1;i>0;i--){const j=Math.floor(Math.random()*(i+1));[starts[i],starts[j]]=[starts[j],starts[i]]}
    const nb=Math.floor(ns2/bs);for(let b=0;b<nb;b++){const rows=starts.slice(b*bs,b*bs+bs);rows.forEach((st,r)=>{for(let k=0;k<ns;k++)cell[st+k]={b,r}});batches.push(rows)}
    ptDraw(cell,batches,ns,`随机抽样：随机偏移 ${off}，切出 ${ns2} 段（每段 ${ns} 个），打乱顺序后每 ${bs} 段组成一个小批量，共 ${nb} 个。\n${batches.map((rows,b)=>`小批量 ${b+1}：两段从 ${rows.join(' 和 ')} 开始`).join('\n')}\n\n相邻的小批量在原序列里不一定挨着，所以每个小批量都要重新初始化隐状态。`)}}
function ptDraw(cell,batches,ns,text){const cols=['--accent','--pos','--neg','--muted'];$('#ptGrid').innerHTML=cell.map((c,i)=>`<div class="pt" style="${c?`background:color-mix(in srgb,var(${cols[c.b%4]}) ${c.r?18:34}%,var(--surface))`:''}"><b>${i}</b><span class="small muted">${c?`批 ${c.b+1}·第 ${c.r+1} 行`:'没用到'}</span></div>`).join('');$('#ptCalc').textContent=text}
$('#ptMode').onchange=ptRender;$('#ptGo').onclick=ptRender;
// ---------- 第 12 天：迷你 DKT 的隐状态 ----------
const RN={ans:[1,1,0,1,1,1,0,0,0,1]};
function rnRun(mem){const wx=[1.1,0.5],Whh=[[0.9*mem,0.25*mem],[-0.1*mem,0.95*mem]],wq=[1.6,0.9],bq=0.1;let h=[0,0];const H=[],P=[];
  const p0=1/(1+Math.exp(-bq));for(const a of RN.ans){const x=a?1:-1;h=[Math.tanh(x*wx[0]+h[0]*Whh[0][0]+h[1]*Whh[1][0]),Math.tanh(x*wx[1]+h[0]*Whh[0][1]+h[1]*Whh[1][1])];H.push(h);P.push(1/(1+Math.exp(-(h[0]*wq[0]+h[1]*wq[1]+bq))))}return{H,P,p0,Whh}}
function rnRender(){const host=$('#rnHost');if(!host.offsetParent)return;const mem=+$('#rnMem').value;$('#rnMemv').textContent=mem.toFixed(2);const r=rnRun(mem),C=colors(),n=RN.ans.length;
  $('#rnRow').innerHTML=RN.ans.map((a,t)=>`<button class="tk ${a?'ok':'no'}" data-t="${t}" aria-label="第 ${t+1} 步${a?'答对':'答错'}"><span class="small muted">t = ${t+1}</span><b>${a?'对':'错'}</b></button>`).join('');
  $$('#rnRow .tk').forEach(b=>b.onclick=()=>{RN.ans[+b.dataset.t]^=1;rnRender()});
  const W=Math.max(260,host.clientWidth),Hh=Math.round(Math.min(260,W*.6)),ctx=fitCanvas($('#rnCv'),W,Hh),pl=34,pr=10,pt=24,pb=24,cw=(W-pl-pr)/n,sx=t=>pl+cw*(t+.5),sy=v=>pt+(1-v)/2*(Hh-pt-pb);
  ctx.strokeStyle=C.line;ctx.lineWidth=1;ctx.font=MONO;ctx.fillStyle=C.muted;ctx.textAlign='right';ctx.textBaseline='middle';for(const v of[-1,0,0.5,1]){ctx.beginPath();ctx.moveTo(pl,sy(v));ctx.lineTo(W-pr,sy(v));ctx.stroke();ctx.fillText(String(v),pl-4,sy(v))}
  r.P.forEach((p,t)=>{ctx.fillStyle=C.pos;ctx.globalAlpha=.35;const y0=sy(0),y1=sy(p);ctx.fillRect(sx(t)-cw*.3,y1,cw*.6,y0-y1);ctx.globalAlpha=1});
  [[0,C.accent,[]],[1,C.ink,[5,4]]].forEach(([k,col,d])=>{ctx.strokeStyle=col;ctx.lineWidth=2;ctx.setLineDash(d);ctx.beginPath();r.H.forEach((h,t)=>{t?ctx.lineTo(sx(t),sy(h[k])):ctx.moveTo(sx(t),sy(h[k]))});ctx.stroke();ctx.setLineDash([])});
  ctx.textAlign='left';ctx.textBaseline='middle';ctx.font=bodyFont(12);[[C.accent,[],'h 第 1 维'],[C.ink,[5,4],'h 第 2 维']].forEach(([c,d,t],i)=>{const x=pl+i*96;ctx.strokeStyle=c;ctx.setLineDash(d);ctx.beginPath();ctx.moveTo(x,10);ctx.lineTo(x+18,10);ctx.stroke();ctx.setLineDash([]);ctx.fillStyle=C.ink;ctx.fillText(t,x+22,10)});ctx.fillStyle=C.pos;ctx.globalAlpha=.5;ctx.fillRect(pl+192,4,14,12);ctx.globalAlpha=1;ctx.fillStyle=C.ink;ctx.fillText('预测下一题答对的概率',pl+212,10);
  const last=r.H[n-1];$('#rnCalc').textContent=`W_hh = ${mem.toFixed(2)} × [[0.90, 0.25], [−0.10, 0.95]]\n\n${r.P.map((p,t)=>`t = ${String(t+1).padStart(2)}  ${RN.ans[t]?'对':'错'}  h = (${r.H[t][0].toFixed(2).padStart(5)}, ${r.H[t][1].toFixed(2).padStart(5)})  下一题答对概率 ${p.toFixed(2)}`).join('\n')}\n\n${mem<0.1?'W_hh ≈ 0：没有记忆，预测只看最后一题的对错。':mem>1?'记忆很强：很久以前的对错也一直影响预测，改一下第 1 步试试。':'改一下前几步的对错，最后一步的预测也会跟着变：这就是隐状态在“记住”历史。'}`}
['#rnAll1','#rnAll0','#rnAlt'].forEach((s,i)=>$(s).onclick=()=>{RN.ans=RN.ans.map((_,t)=>i===0?1:i===1?0:(t+1)%2);rnRender()});$('#rnMem').addEventListener('input',rnRender);
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
// ---------- 第 14 天：门控怎样留住记忆 ----------
function gtRender(){const host=$('#gtHost');if(!host.offsetParent)return;const F=+$('#gtF').value,I=+$('#gtI').value,mode=$('#gtM').value;$('#gtFv').textContent=F.toFixed(2);$('#gtIv').textContent=I.toFixed(2);
  const T=40,noise=Array.from({length:T},(_,t)=>Math.sin(t*2.3)*0.8+Math.cos(t*5.1)*0.4);let c=1,cn=0;const keep=[1],grad=[1];let g=1,h=1,hn=0;
  for(let t=1;t<T;t++){if(mode==='lstm'){c=F*c;cn=F*cn+I*Math.tanh(noise[t]);keep.push(Math.abs(c));g*=F;grad.push(Math.abs(g))}
    else{const hp=Math.tanh(0.9*h+noise[t]),hq=Math.tanh(0.9*hn+noise[t]);g*=0.9*(1-hp*hp);h=hp;hn=hq;keep.push(Math.abs(h-hn));grad.push(Math.abs(g))}}
  const xs=Array.from({length:T},(_,i)=>i+1);host.innerHTML='<canvas id="gtCv" aria-label="记忆随时间的变化"></canvas>';
  drawPlot($('#gtCv'),{series:[{x:xs,y:keep.map(v=>Math.max(v,1e-12)),label:'第 1 步的信息还剩多少'},{x:xs,y:grad.map(v=>Math.max(v,1e-30)),label:'梯度 ∂(第 t 步)/∂(第 1 步)'}],hlines:[],xlabel:'时间步 t',ylabel:'',legend:true,yscale:'log',ylim:[1e-12,2]});
  $('#gtCalc').textContent=mode==='lstm'?`记忆元：Cₜ = F·C_{t−1} + I·C̃ₜ\n第 1 步写入的信息每一步乘 F，40 步后还剩 F³⁹ = ${(F**39).toExponential(2)}\n梯度 ∂C₄₀/∂C₁ = F³⁹，也是 ${(F**39).toExponential(2)}（所以图上两条线重合）\n\n${F>0.98?'遗忘门接近 1：信息和梯度都几乎原样穿过 40 步。':F<0.6?'遗忘门小：很快就忘了，梯度也消失。':'遗忘门适中：慢慢忘掉。'}\n输入门 I 只影响新写入的噪声有多少，不影响旧信息能留多久。\n真实 LSTM 里 F 每一步都是学出来的：需要记住时接近 1，需要清空时接近 0。`:`普通 RNN：hₜ = tanh(0.9·h_{t−1} + xₜ)\n每一步梯度乘 0.9 × tanh′，40 步后约 ${grad[T-1].toExponential(2)}\n第 1 步的不同（h₁ = 1 和 h₁ = 0 两条轨迹的差）也很快被噪声淹没。\n切回 LSTM，把遗忘门调到 0.99 对比一下。`}
['#gtF','#gtI'].forEach(s=>$(s).addEventListener('input',gtRender));$('#gtM').onchange=gtRender;
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
// ---------- 第 19 天：OPE 估计器游乐场 ----------
const OE={seed:5};
function oeSim(){const T=10**+$('#oeT').value,n=+$('#oeN').value,clip=+$('#oeC').value,qe=+$('#oeQ').value;$('#oeTv').textContent=T.toFixed(3);$('#oeNv').textContent=n;$('#oeQv').textContent=qe.toFixed(2);
  const g=sgRng(OE.seed),u=(()=>{let s=OE.seed*977;return()=>{s=s+0x6D2B79F5>>>0;let t=Math.imul(s^s>>>15,1|s);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}})(),K=5;
  const rm=p=>0.3*Math.exp(-(((p-0.6)/0.2)**2)),Q=p=>(1-qe)*rm(p)+qe*(0.6*rm(p)+0.05),sm=a=>{const m=Math.max(...a),e=a.map(v=>Math.exp(v-m)),z=e.reduce((x,y)=>x+y,0);return e.map(v=>v/z)};
  const mu=p=>sm(p.map(v=>8*v)),pi=p=>sm(p.map(v=>-((v-0.7)**2)/T),0);
  let V=0;for(let i=0;i<20000;i++){const p=Array.from({length:K},()=>u()*0.9+0.05),pp=pi(p);V+=pp.reduce((s,v,k)=>s+v*rm(p[k]),0)}V/=20000;
  const R={IPS:[],SNIPS:[],DM:[],DR:[]},ess=[];
  for(let rep=0;rep<200;rep++){let sw=0,sw2=0,swr=0,sdm=0,sdr=0;for(let i=0;i<n;i++){const p=Array.from({length:K},()=>u()*0.9+0.05),m=mu(p),pp=pi(p);let c=u(),a=0;while(a<K-1&&(c-=m[a])>=0)a++;
      const r=rm(p[a])+0.05*g();let w=pp[a]/m[a];if(clip)w=Math.min(w,clip);const dm=pp.reduce((s,v,k)=>s+v*Q(p[k]),0);sw+=w;sw2+=w*w;swr+=w*r;sdm+=dm;sdr+=dm+w*(r-Q(p[a]))}
    R.IPS.push(swr/n);R.SNIPS.push(swr/sw);R.DM.push(sdm/n);R.DR.push(sdr/n);ess.push(sw*sw/sw2/n)}
  OE.res={V,R,ess:ess.reduce((a,b)=>a+b,0)/ess.length,T,clip,qe,n};oeDraw()}
function oeDraw(){const host=$('#oeHost');if(!host.offsetParent||!OE.res)return;const{V,R,ess,T,clip,qe,n}=OE.res,C=colors(),names=['IPS','SNIPS','DM','DR'],cols=[C.neg,C.accent,C.muted,C.pos];
  const all=names.flatMap(k=>R[k]),lo=Math.min(V,...all),hi=Math.max(V,...all),pad=(hi-lo)*0.08||0.01,y0=lo-pad,y1=hi+pad;
  const W=Math.max(260,host.clientWidth),H=260,ctx=fitCanvas($('#oeCv'),W,H),pl=48,pb=24,cw=(W-pl-8)/4,sy=y=>8+(y1-y)/(y1-y0)*(H-8-pb);
  ctx.font=MONO;ctx.fillStyle=C.muted;ctx.textAlign='right';ctx.textBaseline='middle';ctx.strokeStyle=C.line;for(let k=0;k<=4;k++){const y=y0+(y1-y0)*k/4;ctx.beginPath();ctx.moveTo(pl,sy(y));ctx.lineTo(W-8,sy(y));ctx.stroke();ctx.fillText(y.toFixed(3),pl-4,sy(y))}
  ctx.strokeStyle=C.ink;ctx.setLineDash([5,4]);ctx.beginPath();ctx.moveTo(pl,sy(V));ctx.lineTo(W-8,sy(V));ctx.stroke();ctx.setLineDash([]);
  names.forEach((k,i)=>{ctx.fillStyle=cols[i];ctx.globalAlpha=.55;R[k].forEach((v,j)=>{ctx.beginPath();ctx.arc(pl+cw*(i+0.5)+((j*37)%41-20)*cw/60,sy(v),2.2,0,7);ctx.fill()});ctx.globalAlpha=1;ctx.fillStyle=C.ink;ctx.textAlign='center';ctx.textBaseline='alphabetic';ctx.font=bodyFont(12);ctx.fillText(k,pl+cw*(i+0.5),H-6);ctx.font=MONO});
  const st=a=>{const m=a.reduce((x,y)=>x+y,0)/a.length,s=Math.sqrt(a.reduce((x,y)=>x+(y-m)**2,0)/(a.length-1));return[m-V,s]};
  $('#oeCalc').textContent=`真实 V(π) = ${V.toFixed(4)}（虚线）；n = ${n}，温度 ${T.toFixed(3)}，${clip?'截断到 '+clip:'不截断'}，模型误差 ${qe.toFixed(2)}\n平均 ESS = ${ess.toFixed(3)}\n\n         偏差      标准差\n${names.map(k=>{const[b,s]=st(R[k]);return`${k.padEnd(6)} ${(b>=0?'+':'')+b.toFixed(4)}   ${s.toFixed(4)}`}).join('\n')}\n\n${ess<0.15?'ESS 很低：新策略和旧策略差得太远，估计主要靠少数几条记录。':''}${qe===0?'模型完全正确：DM 和 DR 都没有偏差，DR 方差最小。':qe>0.7?'模型误差很大：DM 偏差明显，DR 仍然基本无偏。':''}${clip&&clip<=5?'\n截断很狠：IPS 方差变小，但出现了偏差。':''}`}
$('#oeGo').onclick=()=>{OE.seed++;oeSim()};['#oeT','#oeN','#oeQ'].forEach(s=>$(s).addEventListener('change',oeSim));['#oeT','#oeN','#oeQ'].forEach(s=>$(s).addEventListener('input',()=>{$('#oeTv').textContent=(10**+$('#oeT').value).toFixed(3);$('#oeNv').textContent=$('#oeN').value;$('#oeQv').textContent=(+$('#oeQ').value).toFixed(2)}));$('#oeC').onchange=oeSim;
function oeRender(){if(!OE.res)oeSim();else oeDraw()}
// ---------- 小结勾选、课程表、切换天数 ----------
function checksSetup(){$$('.check-list').forEach(ul=>{const day=ul.dataset.day,key='checks'+day,st=store.get(key,{});$$('input',ul).forEach(c=>{c.checked=!!st[c.dataset.k];c.onchange=()=>{const s=store.get(key,{});s[c.dataset.k]=c.checked;store.set(key,s);checkMsg(ul)}});checkMsg(ul)})}
function checkMsg(ul){const day=ul.dataset.day,all=$$('input',ul),n=all.filter(c=>c.checked).length,msg=ul.parentElement.querySelector('.check-msg');const done=n===all.length;
  msg.innerHTML=done?`<span class="ok">第 ${day} 天完成。</span>${DAYS.includes(String(+day+1))?`<a href="#day${+day+1}">去第 ${+day+1} 天</a>`:'下一天还在准备中。'}`:`已勾 ${n} / ${all.length}。没把握的那条，回到对应的代码格再改几次。`;
  const nav=$(`[data-nav="${day}"]`);if(nav)nav.classList.toggle('done',done)}
const ROAD=[['1','第 1 章 引言 + 2.1 数据操作'],['2','2.2 数据预处理 + 2.3 线性代数'],['3','2.4 微积分 + 2.5 自动微分','paper'],['4','2.6 概率 + 2.7 查阅文档','paper'],['5','3.1 线性回归 + 3.2 从零开始实现','paper'],['6','3.3 简洁实现 + 3.4 softmax 回归','paper'],['7','3.5 至 3.7 图像分类数据集、softmax 回归实现'],
  ['8','4.1 至 4.3 多层感知机（论文里的 FFN 和输出头）','paper'],['9','4.4 至 4.6 模型选择、过拟合、权重衰减、暂退法（论文里的 Dropout、ρ‖g‖²、按验证集选检查点）','paper'],['10','4.8 至 4.9 数值稳定性和初始化、环境和分布偏移（梯度消失；模拟器和真实日志的差别）','paper'],
  ['11','5.1 至 5.2 + 5.5 层和块、参数管理、读写文件（论文的 class DKT(nn.Module)、state_dict 存检查点）','paper'],['12','8.1 + 8.3 至 8.4 序列模型、语言模型、循环神经网络','paper'],['13','8.5 至 8.7 循环神经网络的实现、梯度裁剪、通过时间反向传播（论文的 clip_grad_norm_）','paper'],
  ['14','9.1 至 9.2 GRU 和 LSTM，专题：动手写一个 DKT','paper'],['15','10.1 至 10.3 注意力提示、汇聚、评分函数（DKVMN 的记忆读写）','paper'],['16','10.5 至 10.6 多头注意力、自注意力和位置编码（SAKT）','paper'],['17','10.7 Transformer：残差连接、LayerNorm（simpleKT、AKT）','paper'],
  ['18','11.4 至 11.6 + 11.10 小批量随机梯度下降、动量、Adam（论文用的优化器）','paper'],['19','论文专题：离线策略评估 IPS、SNIPS、DR（书里没有）','paper'],['20','论文专题：逐行读你的 experiment.py','paper'],
  ['选学','4.7 前向传播、反向传播和计算图；4.10 Kaggle 房价预测'],['选学','第 6 至 7 章 卷积神经网络（LeNet、ResNet 等）'],['选学','9.3 至 9.8 深度和双向循环网络、机器翻译、seq2seq'],['选学','第 11 章其余优化算法、第 12 章计算性能'],['选学','第 13 至 15 章 计算机视觉、自然语言处理']];
function roadRender(){$('#roadList').innerHTML=ROAD.map(([d,t,f])=>{const live=DAYS.includes(d);return`<div><span class="d">${/^\d+$/.test(d)?`第 ${d} 天`:d}</span><span>${live?`<a href="#day${d}">${esc(t)}</a>`:esc(t)}${f==='paper'?' <span class="mark">论文</span>':''}</span><span class="st${live?' now':''}">${live?'已上线':'未上线'}</span></div>`}).join('')}
const REDRAW={'1':()=>{kRender();imRender();ofRender()},'2':()=>{dotRender();normRender()},'3':()=>{secRender();gradRender()},'4':()=>{diceRender();bayesRender();bktRender()},'5':()=>{houseRender();if(!sgd.X)sgdData();sgdRender()},'6':()=>{smRender();calRender()},'7':()=>{pixRender()},'8':()=>{uaRender();actRender()},'9':()=>{spRender();rdRender();dpRender()},'10':()=>{sgRender();shRender()},'11':()=>{pcRender()},'12':()=>{tkRender();ptRender();rnRender()},'13':()=>{clRender();btRender()},'14':()=>{gtRender()},'15':()=>{nwRender();sdRender()},'16':()=>{peRender();skRender()},'17':()=>{lnRender();dcRender()},'18':()=>{opRender()},'19':()=>{oeRender()},'20':()=>{}};
function showDay(day,scroll){if(!DAYS.includes(day))day='1';$$('main.day').forEach(m=>m.hidden=m.dataset.day!==day);$$('[data-nav]').forEach(a=>a.classList.toggle('on',a.dataset.nav===day));store.set('lastDay',day);
  if(!RAN[day]){if(day==='7'&&!FM_U8){fmReady.then(ok=>{if(ok&&!RAN['7']){runAll('7');REDRAW['7']()}});}else runAll(day)}$$('#day'+day+' textarea').forEach(autoSize);REDRAW[day]();if(scroll)window.scrollTo(0,0)}
function route(){const h=location.hash.replace('#','');const m=h.match(/^day(\d+)$/);if(m){showDay(m[1],true);return}
  const el=h&&document.getElementById(h);if(el){const main=el.closest('main.day');if(main&&main.hidden){showDay(main.dataset.day,false);setTimeout(()=>el.scrollIntoView(),0)}}}
addEventListener('hashchange',route);

// ---------- 启动 ----------
function redraw(){const d=store.get('lastDay','1');REDRAW[d]&&REDRAW[d]();$$('.plot canvas').forEach(cv=>{if(cv._fig&&cv.offsetParent)drawPlot(cv,cv._fig)})}
let rz;addEventListener('resize',()=>{clearTimeout(rz);rz=setTimeout(redraw,150)});
matchMedia('(prefers-color-scheme: dark)').addEventListener('change',redraw);
new MutationObserver(redraw).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
setupCells();ofData();kqRender();muRender();sqRender();checksSetup();roadRender();
{const h=location.hash.match(/^#day(\d+)$/);showDay(h?h[1]:(location.hash&&document.getElementById(location.hash.slice(1))?.closest('main.day')?.dataset.day)||store.get('lastDay','1'),false);if(location.hash&&!h)route()}
document.fonts&&document.fonts.ready.then(redraw);
