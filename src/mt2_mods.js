
// ---------- 模块 ----------
const fn=(name,call)=>({t:'fn',name,call});
const DT=n=>({t:'dtype',name:n});
const mod=(name,attrs)=>({t:'mod',name,attrs});
function kwOr(kw,args,name,i,def){return kw[name]!==undefined?kw[name]:args[i]!==undefined?args[i]:def}
function dimArg(a,kw,i=0){const v=kw.dim||kw.axis||a[i];if(!v||v.t==='none')return null;if(v.t==='list'||v.t==='tuple')return v.items.map(intOf);return intOf(v)}
const corrArg=kw=>kw.correction&&kw.correction.t!=='none'?numOf(kw.correction):kw.unbiased&&!kw.unbiased.v?0:1;
const keepArg=kw=>!!((kw.keepdim&&kw.keepdim.v)||(kw.keepdims&&kw.keepdims.v));
function makeCat(args,kw){const seq=args[0];if(!seq||!(seq.t==='tuple'||seq.t==='list'))err('TypeError','cat() 的第一个参数要是一组张量，比如 (X, Y)');let dim=kw.dim?intOf(kw.dim):(args[1]?intOf(args[1]):0);
  const ts=seq.items.map(asTensor),nd=ts[0].shape.length;if(dim<0)dim+=nd;if(dim<0||dim>=nd)err('IndexError','Dimension out of range');
  for(const t of ts){if(t.shape.length!==nd||t.shape.some((d,i)=>i!==dim&&d!==ts[0].shape[i]))err('RuntimeError',`Sizes of tensors must match except in dimension ${dim}. Expected size ${ts[0].shape.find((d,i)=>i!==dim)} but got size ${t.shape.find((d,i)=>i!==dim&&d!==ts[0].shape[i])} for tensor number ${ts.indexOf(t)} in the list.`)}
  const out=ts[0].shape.slice();out[dim]=ts.reduce((a,t)=>a+t.shape[dim],0);const outer=prod(out.slice(0,dim)),data=[],srcs=[];
  for(let o=0;o<outer;o++)ts.forEach((t,ti)=>{const chunk=prod(t.shape.slice(dim));for(let i=0;i<chunk;i++){data.push(t.data[o*chunk+i]);srcs.push([ti,o*chunk+i])}});
  let dt='bool';for(const t of ts)if(RANK[t.dtype]>RANK[dt])dt=t.dtype;const R=mkT(out,data,dt);if(nd===2&&prod(out)<=300)R.meta={kind:'cat',parts:ts.map(snapshot),dim};
  return track(R,ts,'CatBackward0',g=>{const gs=ts.map(t=>new Float64Array(t.data.length));srcs.forEach(([ti,k],q)=>gs[ti][k]+=g[q]);return gs})}
function stack(args,kw){const seq=args[0];const dim=kw.dim?intOf(kw.dim):0;const ts=seq.items.map(asTensor);const ex=ts.map(t=>{const s=t.shape.slice();s.splice(dim<0?s.length+1+dim:dim,0,1);return reshapeT(t,s)});return makeCat([tup(ex)],{dim:I(dim)})}
function sizeKw(a,kw){if(kw.size)return dimsArg([kw.size]);return dimsArg(a)}
const noGradCtx={t:'ctx',enter:()=>{const prev=gradEnabled;gradEnabled=false;return()=>{gradEnabled=prev}}};
function multinomialSample(n,probs,shape){const p=[...probs.data],s=p.reduce((a,b)=>a+b,0);const K=p.length,reps=prod(shape),out=new Float64Array(reps*K);
  for(let r=0;r<reps;r++)for(let t=0;t<n;t++){let u=rand()*s,k=0;for(;k<K-1;k++){u-=p[k];if(u<=0)break}out[r*K+k]++}return mkT([...shape,K],out,'float32')}
const torch=mod('torch',{
  float32:DT('float32'),float:DT('float32'),float64:DT('float64'),double:DT('float64'),int64:DT('int64'),long:DT('int64'),int32:DT('int64'),bool:DT('bool'),
  arange:fn('arange',(a,kw)=>{const nums=a.map(numOf);let s=0,e,st=1;if(nums.length===1)e=nums[0];else[s,e,st=1]=nums;if(!a.length)err('TypeError','arange() 至少要一个参数');const fl=a.some(v=>v.t==='float');const len=Math.max(0,Math.ceil((e-s)/st));const d=[];for(let k=0;k<len;k++)d.push(fl?fmaD(k,st,s):s+k*st);return withGrad(mkT([d.length],d,dtypeKw(kw,fl?'float32':'int64')),kw)}),
  zeros:fn('zeros',(a,kw)=>{const s=sizeKw(a,kw);return withGrad(mkT(s,new Array(prod(s)).fill(0),dtypeKw(kw,'float32')),kw)}),
  ones:fn('ones',(a,kw)=>{const s=sizeKw(a,kw);return withGrad(mkT(s,new Array(prod(s)).fill(1),dtypeKw(kw,'float32')),kw)}),
  full:fn('full',(a,kw)=>{const s=dimsArg([a[0]]),v=a[1]||kw.fill_value;return mkT(s,new Array(prod(s)).fill(numOf(v)),dtypeKw(kw,v.t==='float'?'float32':'int64'))}),
  zeros_like:fn('zeros_like',(a,kw)=>{const T=asTensor(a[0]);return mkT(T.shape.slice(),new Array(T.data.length).fill(0),dtypeKw(kw,T.dtype))}),
  ones_like:fn('ones_like',(a,kw)=>{const T=asTensor(a[0]);return mkT(T.shape.slice(),new Array(T.data.length).fill(1),dtypeKw(kw,T.dtype))}),
  randn:fn('randn',(a,kw)=>{const s=sizeKw(a,kw);return withGrad(mkT(s,Array.from({length:prod(s)},gauss),'float32'),kw)}),
  rand:fn('rand',(a,kw)=>{const s=sizeKw(a,kw);return withGrad(mkT(s,Array.from({length:prod(s)},rand),'float32'),kw)}),
  normal:fn('normal',(a,kw)=>{const m=numOf(kwOr(kw,a,'mean',0,F(0))),sd=numOf(kwOr(kw,a,'std',1,F(1))),s=sizeKw([],{size:kwOr(kw,a,'size',2)});return withGrad(mkT(s,Array.from({length:prod(s)},()=>m+sd*gauss()),'float32'),kw)}),
  manual_seed:fn('manual_seed',a=>{seed=intOf(a[0])*2654435761|0;return NONE}),
  tensor:fn('tensor',(a,kw)=>{if(!a.length)err('TypeError','tensor() 需要一个参数');const n=nested(a[0]);let dt=dtypeKw(kw,n.np&&n.dt==='float64'?'float64':n.dt);return withGrad(mkT(n.shape,n.flat,dt),kw)}),
  from_numpy:fn('from_numpy',a=>{const T=asTensor(a[0]);return mkT(T.shape.slice(),T.data,T.dtype)}),
  exp:fn('exp',a=>unary(a[0],'exp')),log:fn('log',a=>unary(a[0],'log')),log2:fn('log2',a=>unary(a[0],'log2')),log10:fn('log10',a=>unary(a[0],'log10')),sqrt:fn('sqrt',a=>unary(a[0],'sqrt')),sin:fn('sin',a=>unary(a[0],'sin')),cos:fn('cos',a=>unary(a[0],'cos')),tanh:fn('tanh',a=>unary(a[0],'tanh')),
  sigmoid:fn('sigmoid',a=>unary(a[0],'sigmoid')),relu:fn('relu',a=>unary(a[0],'relu')),abs:fn('abs',a=>unary(a[0],'abs')),
  cat:fn('cat',makeCat),concat:fn('concat',makeCat),stack:fn('stack',stack),
  matmul:fn('matmul',a=>matmul(asTensor(a[0]),asTensor(a[1]))),mm:fn('mm',a=>{const A=asTensor(a[0]),Bt=asTensor(a[1]);if(A.shape.length!==2||Bt.shape.length!==2)err('RuntimeError',`self must be a matrix`);return matmul(A,Bt)}),
  mv:fn('mv',a=>{const A=asTensor(a[0]),x=asTensor(a[1]);if(A.shape.length!==2||x.shape.length!==1)err('RuntimeError','vector + matrix @ vector expected, got 1, 2, 2');return matmul(A,x,'mv')}),
  dot:fn('dot',a=>{const A=asTensor(a[0]),Bt=asTensor(a[1]);if(A.shape.length!==1||Bt.shape.length!==1)err('RuntimeError',`1D tensors expected, but got ${A.shape.length}D and ${Bt.shape.length}D tensors`);if(A.dtype!==Bt.dtype)err('RuntimeError',`dot : expected both vectors to have same dtype, but found ${A.dtype==='int64'?'Long':'Float'} and ${Bt.dtype==='int64'?'Long':'Float'}`);return matmul(A,Bt)}),
  norm:fn('norm',a=>normT(asTensor(a[0]))),
  sum:fn('sum',(a,kw)=>reduce(asTensor(a[0]),dimArg(a,kw,1),'sum',keepArg(kw))),mean:fn('mean',(a,kw)=>reduce(asTensor(a[0]),dimArg(a,kw,1),'mean',keepArg(kw))),clamp:fn('clamp',(a,kw)=>clampT(asTensor(a[0]),kw.min||a[1],kw.max||a[2])),clip:fn('clip',(a,kw)=>clampT(asTensor(a[0]),kw.min||a[1],kw.max||a[2])),var:fn('var',(a,kw)=>varStd(asTensor(a[0]),dimArg(a,kw,1),corrArg(kw),keepArg(kw),false)),std:fn('std',(a,kw)=>varStd(asTensor(a[0]),dimArg(a,kw,1),corrArg(kw),keepArg(kw),true)),
  max:fn('max',(a,kw)=>a[1]&&isT(a[1])?maxmin(a[0],a[1],1):reduce(asTensor(a[0]),dimArg(a,kw,1),'max',keepArg(kw))),min:fn('min',(a,kw)=>a[1]&&isT(a[1])?maxmin(a[0],a[1],-1):reduce(asTensor(a[0]),dimArg(a,kw,1),'min',keepArg(kw))),
  maximum:fn('maximum',a=>maxmin(a[0],a[1],1)),minimum:fn('minimum',a=>maxmin(a[0],a[1],-1)),
  eye:fn('eye',(a,kw)=>{const n=intOf(a[0]),m=a[1]?intOf(a[1]):n;const d=[];for(let i=0;i<n;i++)for(let j=0;j<m;j++)d.push(i===j?1:0);return mkT([n,m],d,dtypeKw(kw,'float32'))}),
  allclose:fn('allclose',(a,kw)=>{const A=asTensor(a[0]),Bt=asTensor(a[1]);const rt=kw.rtol?numOf(kw.rtol):1e-5,at=kw.atol?numOf(kw.atol):1e-8;const D=binop('-',A,Bt),Bb=binop('*',Bt,I(1));return B([...D.data].every((v,i)=>Math.abs(v)<=at+rt*Math.abs(Bb.data[i%Bb.data.length])))}),
  round:fn('round',a=>{const T=asTensor(a[0]);return mkT(T.shape.slice(),Array.from(T.data,v=>Math.round(v)),T.dtype)}),
  cumsum:fn('cumsum',(a,kw)=>cumsum(asTensor(a[0]),dimArg(a,kw,1))),
  no_grad:fn('no_grad',()=>noGradCtx),
  linalg:mod('torch.linalg',{norm:fn('norm',a=>normT(asTensor(a[0]))),inv:fn('inv',a=>invT(asTensor(a[0]))),solve:fn('solve',a=>matmul(invT(asTensor(a[0])),asTensor(a[1])))}),
  inverse:fn('inverse',a=>invT(asTensor(a[0]))),
  distributions:mod('torch.distributions',{multinomial:mod('torch.distributions.multinomial',{Multinomial:fn('Multinomial',(a,kw)=>({t:'dist',name:'Multinomial',n:intOf(a[0]),probs:asTensor(a[1]||kw.probs)}))})}),
  Tensor:{t:'type',name:'torch.Tensor'},
});
torch.attrs.distributions.attrs.Multinomial=torch.attrs.distributions.attrs.multinomial.attrs.Multinomial;
function distAttr(d,name){if(name==='sample')return fn('sample',a=>multinomialSample(d.n,d.probs,a[0]?dimsArg([a[0]]):[]));if(name==='probs')return d.probs;err('AttributeError',`'Multinomial' object has no attribute '${name}'`)}
const npArr=(T)=>{const R=mkT(T.shape.slice(),T.data,T.dtype,true);return R};
const numpy=mod('numpy',{
  arange:fn('arange',(a,kw)=>{const nums=a.map(numOf);let s=0,e,st=1;if(nums.length===1)e=nums[0];else[s,e,st=1]=nums;const fl=a.some(v=>v.t==='float');const d=[];for(let k=0;;k++){const v=s+k*st;if(st>0?v>=e-1e-12:v<=e+1e-12)break;d.push(v)}return mkT([d.length],d,fl?'float64':'int64',true)}),
  array:fn('array',a=>{const n=nested(a[0]);return mkT(n.shape,n.flat,n.dt==='float32'?'float64':n.dt,true)}),
  linspace:fn('linspace',a=>{const s=numOf(a[0]),e=numOf(a[1]),n=a[2]?intOf(a[2]):50;return mkT([n],Array.from({length:n},(_,i)=>n===1?s:s+(e-s)*i/(n-1)),'float64',true)}),
  ...Object.fromEntries(['exp','log','sqrt','sin','cos','abs','tanh'].map(n=>[n,fn(n,a=>{const v=a[0];if(!isT(v)&&pyNum(v)!=null){const r=F(UN[n][0](pyNum(v)));r.np=true;return r}return unary(isT(v)?v:asTensor(v),n)})])),
  zeros:fn('zeros',a=>{const s=dimsArg([a[0]]);return mkT(s,new Array(prod(s)).fill(0),'float64',true)}),ones:fn('ones',a=>{const s=dimsArg([a[0]]);return mkT(s,new Array(prod(s)).fill(1),'float64',true)}),
  power:fn('power',a=>binop('**',asTensor(a[0]),asTensor(a[1]))),dot:fn('dot',a=>matmul(asTensor(a[0]),asTensor(a[1]))),sum:fn('sum',(a,kw)=>reduce(asTensor(a[0]),dimArg(a,kw,1),'sum',keepArg(kw))),mean:fn('mean',(a,kw)=>reduce(asTensor(a[0]),dimArg(a,kw,1),'mean',keepArg(kw))),
  pi:F(Math.PI),e:F(Math.E),nan:F(NaN),float32:DT('float32'),float64:DT('float64'),
  random:mod('numpy.random',{normal:fn('normal',(a,kw)=>{const m=numOf(kwOr(kw,a,'loc',0,F(0))),sd=numOf(kwOr(kw,a,'scale',1,F(1))),s=dimsArg([kwOr(kw,a,'size',2,tup([]))]);return mkT(s,Array.from({length:prod(s)},()=>m+sd*gauss()),'float64',true)}),rand:fn('rand',a=>{const s=dimsArg(a);return mkT(s,Array.from({length:prod(s)},rand),'float64',true)}),
    shuffle:fn('shuffle',a=>{const T=a[0];if(isT(T)){const n=T.shape[0],row=T.data.length/n,d=T.data;for(let i=n-1;i>0;i--){const j=Math.floor(rand()*(i+1));for(let k=0;k<row;k++){const t=d[i*row+k];d[i*row+k]=d[j*row+k];d[j*row+k]=t}}}else{const L=a[0].items;for(let i=L.length-1;i>0;i--){const j=Math.floor(rand()*(i+1));[L[i],L[j]]=[L[j],L[i]]}}return NONE}),seed:fn('seed',a=>{seed=intOf(a[0])*2654435761|0;return NONE})}),
});
const pandas=mod('pandas',{read_csv:fn('read_csv',a=>readCSV(str(a[0]))),get_dummies:fn('get_dummies',(a,kw)=>getDummies(a[0],kw)),DataFrame:fn('DataFrame',a=>a[0].t==='dict'?dfFromDict(a[0]):err('TypeError','本页的 DataFrame 只支持用字典创建')),
  Series:fn('Series',(a,kw)=>{const c=pyListCol(iterate(a[0]));return mkSeries(c.v,c.dt,c.v.map((_,i)=>i),kw.name?str(kw.name):null)}),isna:fn('isna',a=>dfAttr(a[0],'isna').call([],{})),NA:F(NaN)});
const os=mod('os',{makedirs:fn('makedirs',(a,kw)=>{const p=normPath(str(a[0]));if(DIRS.has(p)&&!(kw.exist_ok&&kw.exist_ok.v))err('FileExistsError',`[Errno 17] File exists: '${str(a[0])}'`);let cur='';for(const part of p.split('/')){cur=cur?cur+'/'+part:part;DIRS.add(cur)}return NONE}),
  listdir:fn('listdir',a=>{const p=normPath(a[0]?str(a[0]):'.');return list([...FS.keys()].filter(k=>(k.includes('/')?k.slice(0,k.lastIndexOf('/')):'.')===p).map(k=>S(k.split('/').pop())))}),
  path:mod('os.path',{join:fn('join',a=>S(a.map(str).join('/'))),exists:fn('exists',a=>{const p=normPath(str(a[0]));return B(FS.has(p)||DIRS.has(p))})}),getcwd:fn('getcwd',()=>S('/home/d2l/notebooks'))});
const math=mod('math',{sqrt:fn('sqrt',a=>F(Math.sqrt(numOf(a[0])))),exp:fn('exp',a=>F(Math.exp(numOf(a[0])))),log:fn('log',a=>F(a[1]?Math.log(numOf(a[0]))/Math.log(numOf(a[1])):Math.log(numOf(a[0])))),
  sin:fn('sin',a=>F(Math.sin(numOf(a[0])))),cos:fn('cos',a=>F(Math.cos(numOf(a[0])))),pi:F(Math.PI),e:F(Math.E),inf:F(Infinity),floor:fn('floor',a=>I(Math.floor(numOf(a[0])))),ceil:fn('ceil',a=>I(Math.ceil(numOf(a[0])))),gamma:fn('gamma',a=>F(gammaFn(numOf(a[0])))),factorial:fn('factorial',a=>{let r=1;for(let i=2;i<=intOf(a[0]);i++)r*=i;return I(r)})});
const random=mod('random',{random:fn('random',()=>F(rand())),seed:fn('seed',a=>{seed=intOf(a[0])*2654435761|0;return NONE}),randint:fn('randint',a=>{const lo=intOf(a[0]),hi=intOf(a[1]);return I(lo+Math.floor(rand()*(hi-lo+1)))}),
  shuffle:fn('shuffle',a=>{const L=a[0].items;for(let i=L.length-1;i>0;i--){const j=Math.floor(rand()*(i+1));[L[i],L[j]]=[L[j],L[i]]}return NONE}),choice:fn('choice',a=>{const L=iterate(a[0]);return L[Math.floor(rand()*L.length)]}),
  uniform:fn('uniform',a=>F(numOf(a[0])+rand()*(numOf(a[1])-numOf(a[0])))),gauss:fn('gauss',a=>F(numOf(a[0])+numOf(a[1])*gauss()))});
// d2l：画图函数直接在页面里画，不走 matplotlib
function toXY(v){if(isT(v))return[...v.data];if(v.items)return v.items.map(numOf);err('TypeError','画图的数据要是张量或列表')}
function plotFig(ctx){if(!ctx.fig){ctx.fig={series:[],hlines:[],xlabel:'',ylabel:'',legend:false};ctx.plots.push(ctx.fig)}return ctx.fig}
function d2lPlot(a,kw,ctx){let X=a[0],Y=kwOr(kw,a,'Y',1,null);const xl=kwOr(kw,a,'xlabel',2,NONE),yl=kwOr(kw,a,'ylabel',3,NONE),lg=kwOr(kw,a,'legend',4,NONE);
  const isList=v=>v&&(v.t==='list'||v.t==='tuple')&&v.items.length&&(isT(v.items[0])||v.items[0].t==='list');
  let Xs=isList(X)?X.items:[X],Ys;if(!Y||Y.t==='none'){Ys=Xs;Xs=Ys.map(y=>null)}else Ys=isList(Y)?Y.items:[Y];if(Xs.length===1&&Ys.length>1)Xs=Ys.map(()=>Xs[0]);
  const labels=lg&&lg.t!=='none'?lg.items.map(str):[];const fig={series:Ys.map((y,i)=>{const yv=toXY(y),xv=Xs[i]?toXY(Xs[i]):yv.map((_,k)=>k);return{x:xv,y:yv,label:labels[i]||''}}),hlines:[],xlabel:xl.t==='str'?xl.v:'',ylabel:yl.t==='str'?yl.v:'',legend:labels.length>0};
  ctx.plots.push(fig);return NONE}
const pltAxes={t:'mod',name:'Axes',attrs:{}};
function pltMod(ctx){return{t:'mod',name:'matplotlib.pyplot',attrs:new Proxy({},{get:(o,name)=>{
  const P={plot:(a,kw)=>{const f=plotFig(ctx);let x,y;if(a.length>=2&&a[1].t!=='str'){x=toXY(a[0]);y=toXY(a[1])}else{y=toXY(a[0]);x=y.map((_,i)=>i)}const lab=kw.label?str(kw.label):'';if(lab)f.legend=true;f.series.push({x,y,label:lab});return NONE},
    scatter:(a,kw)=>{const f=plotFig(ctx);f.series.push({x:toXY(a[0]),y:toXY(a[1]),label:kw.label?str(kw.label):'',scatter:true});return NONE},
    axhline:(a,kw)=>{plotFig(ctx).hlines.push(numOf(kw.y||a[0]||F(0)));return NONE},xlabel:a=>{plotFig(ctx).xlabel=str(a[0]);return NONE},ylabel:a=>{plotFig(ctx).ylabel=str(a[0]);return NONE},legend:()=>{plotFig(ctx).legend=true;return NONE},
    gca:()=>({t:'mod',name:'Axes',attrs:{set_xlabel:fn('set_xlabel',a=>{plotFig(ctx).xlabel=str(a[0]);return NONE}),set_ylabel:fn('set_ylabel',a=>{plotFig(ctx).ylabel=str(a[0]);return NONE}),set_title:fn('set_title',()=>NONE)}}),
    subplots:(a,kw)=>subplots(a,kw),show:()=>NONE,title:()=>NONE,figure:()=>NONE,grid:()=>NONE,rcParams:DUMMY};
  return P[name]?fn(name,P[name]):DUMMY}})}}
function makeTimer(){const o={times:[],tik:0};const now=()=>(typeof performance!=='undefined'?performance.now():Date.now())/1000;o.tik=now();
  return{t:'mod',name:'Timer',attrs:{times:{get t(){return'list'},get items(){return o.times.map(F)}},start:fn('start',()=>{o.tik=now();return NONE}),stop:fn('stop',()=>{o.times.push(now()-o.tik);return F(o.times[o.times.length-1])}),
    avg:fn('avg',()=>F(o.times.reduce((a,b)=>a+b,0)/o.times.length)),sum:fn('sum',()=>F(o.times.reduce((a,b)=>a+b,0))),cumsum:fn('cumsum',()=>{let s=0;return list(o.times.map(v=>F(s+=v)))})}}}
const D2L_ALIAS={ones:'ones',zeros:'zeros',tensor:'tensor',normal:'normal',matmul:'matmul',arange:'arange',exp:'exp',sin:'sin',cos:'cos',tanh:'tanh',abs:'abs',float32:'float32',int64:'int64',concat:'cat',stack:'stack',randn:'randn',rand:'rand',zeros_like:'zeros_like',ones_like:'ones_like',sigmoid:'sigmoid',log:'log',sqrt:'sqrt'};
function d2lMod(ctx){return mod('d2l',{
  ...Object.fromEntries(Object.entries(D2L_ALIAS).map(([k,v])=>[k,torch.attrs[v]])),
  reshape:fn('reshape',a=>reshapeT(asTensor(a[0]),dimsArg([a[1]]))),numpy:fn('numpy',a=>{const T=asTensor(a[0]);const R=mkT(T.shape.slice(),[],T.dtype,true);R.data=T.data;return R}),
  reduce_sum:fn('reduce_sum',(a,kw)=>reduce(asTensor(a[0]),dimArg(a,kw,1),'sum',keepArg(kw))),size:fn('size',a=>I(asTensor(a[0]).data.length)),
  Timer:fn('Timer',()=>makeTimer()),
  linreg:fn('linreg',a=>binop('+',matmul(asTensor(a[0]),asTensor(a[1])),a[2])),
  squared_loss:fn('squared_loss',a=>{const yh=asTensor(a[0]);return binop('/',binop('**',binop('-',yh,reshapeT(asTensor(a[1]),yh.shape.slice())),I(2)),I(2))}),
  sgd:fn('sgd',a=>{const prev=gradEnabled;gradEnabled=false;try{for(const p of iterate(a[0])){const r=binop('-',p,binop('/',binop('*',a[1],p.grad),a[2]));p.data.set(r.data.map(v=>castVal(v,p.dtype)));p.grad.data.fill(0)}}finally{gradEnabled=prev}return NONE}),plot:fn('plot',(a,kw)=>d2lPlot(a,kw,ctx)),set_figsize:fn('set_figsize',()=>NONE),use_svg_display:fn('use_svg_display',()=>NONE),plt:pltMod(ctx),torch:torch,
  synthetic_data:fn('synthetic_data',a=>{const w=asTensor(a[0]),b=numOf(a[1]),n=intOf(a[2]);const X=mkT([n,w.shape[0]],Array.from({length:n*w.shape[0]},gauss),'float32');let y=matmul(X,w);y=binop('+',y,F(b));const noise=mkT(y.shape.slice(),Array.from({length:y.data.length},()=>gauss()*0.01),'float32');y=binop('+',y,noise);return tup([X,reshapeT(y,[-1,1])])})})}
