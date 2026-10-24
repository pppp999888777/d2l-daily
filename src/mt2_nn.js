
// ---------- 神经网络：nn 层、损失、优化器、数据加载 ----------
function mkParam(shape,data){const p=mkT(shape,data,'float32');p.requires_grad=true;p.isParam=true;return p}
function rename(T,name){if(T.grad_fn)T.grad_fn.name=name;return T}
// 按元素的自定义运算（带自动求导）
function ew(x,f,df,name,dt){const X=asTensor(x);const out=Float64Array.from(X.data,f);const R=mkT(X.shape.slice(),out,dt||(isFloatDt(X.dtype)?X.dtype:'float32'),X.np);return track(R,[X],name,g=>[Float64Array.from(g,(gv,k)=>gv*df(X.data[k],R.data[k]))])}
function groups(T,dim){const nd=T.shape.length;if(dim<0)dim+=nd;if(dim<0||dim>=nd)err('IndexError',`Dimension out of range (expected to be in range of [${-nd}, ${nd-1}], but got ${dim})`);const st=strides(T.shape),n=T.shape[dim],outer=T.data.length/n,res=[];
  for(let o=0;o<outer;o++){const hi=Math.floor(o/st[dim]),lo=o%st[dim],base=hi*st[dim]*n+lo;res.push(Array.from({length:n},(_,i)=>base+i*st[dim]))}return res}
function softmaxT(x,dim,log){const X=asTensor(x);if(!isFloatDt(X.dtype))err('RuntimeError',`"softmax_lastdim_kernel_impl" not implemented for 'Long'`);const G=groups(X,dim);const out=new Float64Array(X.data.length);
  for(const g of G){let m=-Infinity;for(const i of g)m=Math.max(m,X.data[i]);let s=0;for(const i of g)s+=Math.exp(X.data[i]-m);for(const i of g)out[i]=log?X.data[i]-m-Math.log(s):Math.exp(X.data[i]-m)/s}
  const R=mkT(X.shape.slice(),out,X.dtype);return track(R,[X],log?'LogSoftmaxBackward0':'SoftmaxBackward0',gr=>{const r=new Float64Array(gr.length);for(const g of G){if(log){let sg=0;for(const i of g)sg+=gr[i];for(const i of g)r[i]=gr[i]-Math.exp(out[i])*sg}else{let d=0;for(const i of g)d+=gr[i]*out[i];for(const i of g)r[i]=out[i]*(gr[i]-d)}}return[r]})}
function reduceLoss(L,red,name){if(red==='none')return rename(L,name);const R=reduce(L,null,red==='sum'?'sum':'mean',false);return rename(R,name)}
const redOf=kw=>kw.reduction?str(kw.reduction):'mean';
function mseLoss(a,b,red){const A=asTensor(a),Bt=asTensor(b);if(A.shape.join()!==Bt.shape.join()&&A.data.length!==Bt.data.length)warnings.push(`UserWarning: Using a target size (torch.Size([${Bt.shape.join(', ')}])) that is different to the input size (torch.Size([${A.shape.join(', ')}])). This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.`);
  return reduceLoss(binop('**',binop('-',A,Bt),I(2)),red,'MseLossBackward0')}
function huberLoss(a,b,red,delta){const D=binop('-',asTensor(a),asTensor(b));const L=ew(D,d=>Math.abs(d)<=delta?0.5*d*d:delta*(Math.abs(d)-0.5*delta),d=>Math.abs(d)<=delta?d:delta*Math.sign(d),'HuberLossBackward0');return reduceLoss(L,red,'HuberLossBackward0')}
function smoothL1(a,b,red,beta){const D=binop('-',asTensor(a),asTensor(b));const L=ew(D,d=>Math.abs(d)<beta?0.5*d*d/beta:Math.abs(d)-0.5*beta,d=>Math.abs(d)<beta?d/beta:Math.sign(d),'SmoothL1LossBackward0');return reduceLoss(L,red,'SmoothL1LossBackward0')}
function crossEntropy(logits,target,red,ls){const X=asTensor(logits),Y=asTensor(target);if(X.shape.length===1)return reshapeT(crossEntropy(reshapeT(X,[1,X.shape[0]]),reshapeT(Y,[1].concat(Y.shape.length?[Y.shape[0]]:[])),'none'),[]);
  const[N,C]=X.shape;const isIdx=!isFloatDt(Y.dtype);if(isIdx&&Y.shape.length!==1)err('RuntimeError','1D target tensor expected, multi-target not supported');if(isIdx)for(const t of Y.data)if(t<0||t>=C)err('IndexError',`Target ${t} is out of bounds.`);
  const lsm=new Float64Array(N*C),out=new Float64Array(N);for(let i=0;i<N;i++){let m=-Infinity;for(let j=0;j<C;j++)m=Math.max(m,X.data[i*C+j]);let s=0;for(let j=0;j<C;j++)s+=Math.exp(X.data[i*C+j]-m);for(let j=0;j<C;j++)lsm[i*C+j]=X.data[i*C+j]-m-Math.log(s);
    out[i]=isIdx?-lsm[i*C+Y.data[i]]:-[...Array(C).keys()].reduce((a,j)=>a+Y.data[i*C+j]*lsm[i*C+j],0)}
  const L=mkT([N],out,X.dtype);track(L,[X],'NllLossBackward0',g=>{const r=new Float64Array(N*C);for(let i=0;i<N;i++)for(let j=0;j<C;j++){const p=Math.exp(lsm[i*C+j]);const y=isIdx?(Y.data[i]===j?1:0):Y.data[i*C+j];r[i*C+j]=g[i]*(p-y)}return[r]});
  return reduceLoss(L,red,'NllLossBackward0')}
function bceLoss(p,y,red,logits){const P=asTensor(p),Y=asTensor(y);if(P.shape.join()!==Y.shape.join())err('ValueError',`Using a target size (torch.Size([${Y.shape.join(', ')}])) that is different to the input size (torch.Size([${P.shape.join(', ')}])) is deprecated. Please ensure they have the same size.`);
  const n=P.data.length,out=new Float64Array(n);for(let k=0;k<n;k++){const y=Y.data[k],z=P.data[k];if(logits)out[k]=Math.max(z,0)-z*y+Math.log(1+Math.exp(-Math.abs(z)));else{const q=Math.min(Math.max(z,0),1);out[k]=-(y*Math.max(Math.log(q),-100)+(1-y)*Math.max(Math.log(1-q),-100))}}
  const nm=logits?'BinaryCrossEntropyWithLogitsBackward0':'BinaryCrossEntropyBackward0';const L=mkT(P.shape.slice(),out,P.dtype);
  track(L,[P],nm,g=>[Float64Array.from(g,(gv,k)=>{const y=Y.data[k],z=P.data[k];return logits?gv*(1/(1+Math.exp(-z))-y):gv*(z-y)/Math.max(z*(1-z),1e-12)})]);return reduceLoss(L,red,nm)}
let warnings=[];
// 层
let nnTraining=true;
function nnObj(kind,cfg,params,children){return{t:'nn',kind,cfg:cfg||{},params:params||{},children:children||[],training:true}}
function linearInit(inF,outF,bias){const k=1/Math.sqrt(inF),u=()=>(rand()*2-1)*k;const p={weight:mkParam([outF,inF],Array.from({length:outF*inF},u))};p.bias=bias?mkParam([outF],Array.from({length:outF},u)):null;return p}
// ---------- 循环层：RNN / LSTM / GRU（用可求导的基本运算一步步算，自动求导自然成立） ----------
const RNN_GATES={RNN:1,LSTM:4,GRU:3};
function rnnMake(kind,a,kw){const inS=intOf(kwOr(kw,a,'input_size',0)),H=intOf(kwOr(kw,a,'hidden_size',1)),L=intOf(kwOr(kw,a,'num_layers',2,I(1))),bias=kw.bias?truthy(kw.bias):true,bf=kw.batch_first?truthy(kw.batch_first):false;
  if(kw.bidirectional&&truthy(kw.bidirectional))err('NotImplementedError','本页的循环层还不支持 bidirectional=True');
  const G=RNN_GATES[kind],k=1/Math.sqrt(H),u=n=>Array.from({length:n},()=>(rand()*2-1)*k),p={};
  for(let l=0;l<L;l++){const d=l?H:inS;p['weight_ih_l'+l]=mkParam([G*H,d],u(G*H*d));p['weight_hh_l'+l]=mkParam([G*H,H],u(G*H*H));if(bias){p['bias_ih_l'+l]=mkParam([G*H],u(G*H));p['bias_hh_l'+l]=mkParam([G*H],u(G*H))}}
  const nl=kw.nonlinearity?str(kw.nonlinearity):'tanh';
  return nnObj(kind,{input_size:inS,hidden_size:H,num_layers:L,bias,batch_first:bf,nonlinearity:nl,attrs:{input_size:I(inS),hidden_size:I(H),num_layers:I(L),bidirectional:B(false),batch_first:B(bf),bias:B(bias)}},p)}
// 融合的单步计算：一个时间步只建一个计算图节点，反向传播手写（公式见第 13、14 天）。比拆成二十几个小运算快很多。
const sigm=x=>1/(1+Math.exp(-x));
function mmT(A,W,B,n,k,G){const out=new Float64Array(B*G);for(let b=0;b<B;b++)for(let g=0;g<G;g++){let s=0;const wo=g*n,ao=b*n;for(let j=0;j<n;j++)s+=A[ao+j]*W[wo+j];out[b*G+g]=s}return out} // A(B,n) · Wᵀ，W 是 (G,n)
function cellBack(dg,h,Wh,B,H,G,hasB){const gW=new Float64Array(G*H),gb=hasB?new Float64Array(G):null,gh=new Float64Array(B*H);
  for(let b=0;b<B;b++)for(let g=0;g<G;g++){const d=dg[b*G+g];if(!d)continue;if(gb)gb[g]+=d;const wo=g*H,ho=b*H;for(let j=0;j<H;j++){gW[wo+j]+=d*h[ho+j];gh[ho+j]+=d*Wh[wo+j]}}return{gW,gb,gh}}
function fusedCell(kind,gi,h,c,Wh,bh,relu){const B=h.shape[0],H=h.shape[1],G=Wh.shape[0],hd=h.data,ghh=mmT(hd,Wh.data,B,H,H,G);if(bh)for(let b=0;b<B;b++)for(let g=0;g<G;g++)ghh[b*G+g]+=bh.data[g];
  const ins=kind==='LSTM'?[gi,h,c,Wh,bh]:[gi,h,Wh,bh],gd=gi.data;
  if(kind==='RNN'){const out=new Float64Array(B*H);for(let q=0;q<B*H;q++){const v=gd[q]+ghh[q];out[q]=relu?Math.max(0,v):Math.tanh(v)}const R=mkT([B,H],out,h.dtype);
    return track(R,ins,'TanhBackward0',g=>{const dp=new Float64Array(B*H);for(let q=0;q<B*H;q++)dp[q]=relu?(R.data[q]>0?g[q]:0):g[q]*(1-R.data[q]*R.data[q]);const r=cellBack(dp,hd,Wh.data,B,H,G,!!bh);return[dp,r.gh,r.gW,r.gb]})}
  if(kind==='GRU'){const rr=new Float64Array(B*H),zz=new Float64Array(B*H),nn=new Float64Array(B*H),out=new Float64Array(B*H);
    for(let b=0;b<B;b++)for(let j=0;j<H;j++){const o=b*G,q=b*H+j,r=sigm(gd[o+j]+ghh[o+j]),z=sigm(gd[o+H+j]+ghh[o+H+j]),n=Math.tanh(gd[o+2*H+j]+r*ghh[o+2*H+j]);rr[q]=r;zz[q]=z;nn[q]=n;out[q]=(1-z)*n+z*hd[q]}
    return track(mkT([B,H],out,h.dtype),ins,'AddBackward0',g=>{const dgi=new Float64Array(B*G),dgh=new Float64Array(B*G),dh=new Float64Array(B*H);
      for(let b=0;b<B;b++)for(let j=0;j<H;j++){const q=b*H+j,o=b*G,r=rr[q],z=zz[q],n=nn[q],gv=g[q];const dn=gv*(1-z)*(1-n*n),dz=gv*(hd[q]-n)*z*(1-z),dr=dn*ghh[o+2*H+j]*r*(1-r);dh[q]=gv*z;
        dgi[o+j]=dr;dgh[o+j]=dr;dgi[o+H+j]=dz;dgh[o+H+j]=dz;dgi[o+2*H+j]=dn;dgh[o+2*H+j]=dn*r}
      const r=cellBack(dgh,hd,Wh.data,B,H,G,!!bh);for(let q=0;q<B*H;q++)dh[q]+=r.gh[q];return[dgi,dh,r.gW,r.gb]})}
  // LSTM：输出把 h 和 c 拼在一起 (B, 2H)，外面再切开
  const cd=c.data,I_=new Float64Array(B*H),F_=new Float64Array(B*H),Gg=new Float64Array(B*H),O_=new Float64Array(B*H),TC=new Float64Array(B*H),out=new Float64Array(B*2*H);
  for(let b=0;b<B;b++)for(let j=0;j<H;j++){const o=b*G,q=b*H+j,i=sigm(gd[o+j]+ghh[o+j]),f=sigm(gd[o+H+j]+ghh[o+H+j]),gg=Math.tanh(gd[o+2*H+j]+ghh[o+2*H+j]),oo=sigm(gd[o+3*H+j]+ghh[o+3*H+j]);
    const cn=f*cd[q]+i*gg,tc=Math.tanh(cn);I_[q]=i;F_[q]=f;Gg[q]=gg;O_[q]=oo;TC[q]=tc;out[b*2*H+j]=oo*tc;out[b*2*H+H+j]=cn}
  return track(mkT([B,2*H],out,h.dtype),ins,'CatBackward0',g=>{const dg=new Float64Array(B*G),dc=new Float64Array(B*H);
    for(let b=0;b<B;b++)for(let j=0;j<H;j++){const q=b*H+j,o=b*G,dhv=g[b*2*H+j],i=I_[q],f=F_[q],gg=Gg[q],oo=O_[q],tc=TC[q];const dcn=g[b*2*H+H+j]+dhv*oo*(1-tc*tc);
      dg[o+j]=dcn*gg*i*(1-i);dg[o+H+j]=dcn*cd[q]*f*(1-f);dg[o+2*H+j]=dcn*i*(1-gg*gg);dg[o+3*H+j]=dhv*tc*oo*(1-oo);dc[q]=dcn*f}
    const r=cellBack(dg,hd,Wh.data,B,H,G,!!bh);return[dg,r.gh,dc,r.gW,r.gb]})}
function rnnForward(m,args){const c=m.cfg,H=c.hidden_size,L=c.num_layers,kind=m.kind;let X=asTensor(args[0]);const unb=X.shape.length===2;
  if(X.shape.length!==3&&!unb)err('RuntimeError',`${kind}: Expected input to be 2D or 3D, got ${X.shape.length}D instead`);
  if(unb)X=reshapeT(X,c.batch_first?[1,...X.shape]:[X.shape[0],1,X.shape[1]]);
  if(X.shape[2]!==c.input_size)err('RuntimeError',`input.size(-1) must be equal to input_size. Expected ${c.input_size}, got ${X.shape[2]}`);
  const T=c.batch_first?X.shape[1]:X.shape[0],Bn=c.batch_first?X.shape[0]:X.shape[1],ALL={slice:[null,null,null]};
  let h0=null,c0=null,st=args[1];if(st&&st.t!=='none'){if(kind==='LSTM'){if(st.t!=='tuple'||st.items.length!==2)err('RuntimeError','LSTM 的状态要是 (h0, c0) 两个张量组成的元组');h0=asTensor(st.items[0]);c0=asTensor(st.items[1])}else h0=asTensor(st);if(unb){h0=reshapeT(h0,[L,1,H]);if(c0)c0=reshapeT(c0,[L,1,H])}
    if(h0.shape.join()!==[L,Bn,H].join())err('RuntimeError',`Expected hidden size (${L}, ${Bn}, ${H}), got [${h0.shape.join(', ')}]`)}
  const zeros=()=>mkT([Bn,H],new Float64Array(Bn*H),'float32'),sl=(G,k)=>getItem(G,[ALL,{slice:[k*H,(k+1)*H,null]}]);
  // 加速：权重的转置每层只做一次；输入那一半 X·W_ihᵀ 对所有时间步一次算完，再按时间步取出来
  let inputs=null;const hn=[],cn=[];
  for(let l=0;l<L;l++){const P=m.params,WiT=transpose2(P['weight_ih_l'+l]),bi=P['bias_ih_l'+l],bh=P['bias_hh_l'+l],G=WiT.shape[1];let h=h0?getItem(h0,[{v:l}]):zeros(),cc=kind==='LSTM'?(c0?getItem(c0,[{v:l}]):zeros()):null;const outs=[];
    let gAll;if(l===0){const X2=reshapeT(X,[-1,X.shape[2]]);let g=matmul(X2,WiT);if(bi)g=binop('+',g,bi);gAll=reshapeT(g,[X.shape[0],X.shape[1],G])}else{let g=matmul(reshapeT(stack([list(inputs)],{dim:I(0)}),[-1,H]),WiT);if(bi)g=binop('+',g,bi);gAll=reshapeT(g,[T,Bn,G])}
    const tm=l>0||!c.batch_first;
    for(let t=0;t<T;t++){const gi=tm?getItem(gAll,[{v:t}]):getItem(gAll,[ALL,{v:t}]);
      if(kind==='LSTM'){const hc=fusedCell('LSTM',gi,h,cc,P['weight_hh_l'+l],bh);h=getItem(hc,[ALL,{slice:[0,H,null]}]);cc=getItem(hc,[ALL,{slice:[H,2*H,null]}])}
      else h=fusedCell(kind,gi,h,null,P['weight_hh_l'+l],bh,c.nonlinearity==='relu');
      outs.push(h)}
    inputs=outs;hn.push(h);if(cc)cn.push(cc)}
  const stk=(ts,d)=>rename(stack([list(ts)],{dim:I(d)}),'StackBackward0');
  let out=c.batch_first?rename(stack([list(inputs)],{dim:I(1)}),'TransposeBackward0'):stk(inputs,0),hN=stk(hn,0),cN=cn.length?stk(cn,0):null;
  if(unb){out=reshapeT(out,[T,H]);hN=reshapeT(hN,[L,H]);if(cN)cN=reshapeT(cN,[L,H])}
  return tup([out,kind==='LSTM'?tup([hN,cN]):hN])}
function embedForward(m,x){const I0=asTensor(x),W=m.params.weight,[N,D]=W.shape;if(I0.dtype!=='int64')err('RuntimeError',"Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int; but got torch.FloatTensor instead (while checking arguments for embedding)");
  const out=new Float64Array(I0.data.length*D);I0.data.forEach((v,i)=>{if(v<0||v>=N)err('IndexError','index out of range in self');for(let j=0;j<D;j++)out[i*D+j]=W.data[v*D+j]});
  return track(mkT([...I0.shape,D],out,W.dtype),[W],'EmbeddingBackward0',g=>{const gw=new Float64Array(N*D);I0.data.forEach((v,i)=>{for(let j=0;j<D;j++)gw[v*D+j]+=g[i*D+j]});return[gw]})}
function nnForward(m,args,ctx){const x=args[0];
  switch(m.kind){
    case'RNN':case'LSTM':case'GRU':return rnnForward(m,args);case'Embedding':return embedForward(m,x);
    case'Linear':{const X=asTensor(x),W=m.params.weight,b=m.params.bias;if(X.shape[X.shape.length-1]!==W.shape[1])err('RuntimeError',`mat1 and mat2 shapes cannot be multiplied (${X.shape.length===1?1:prod(X.shape.slice(0,-1))}x${X.shape[X.shape.length-1]} and ${W.shape[1]}x${W.shape[0]})`);
      let Xf=X.shape.length>2?reshapeT(X,[-1,X.shape[X.shape.length-1]]):X;let R=matmul(Xf,transpose2(W));if(b)R=binop('+',R,b);if(X.shape.length>2){R=reshapeT(R,[...X.shape.slice(0,-1),W.shape[0]]);if(!b)rename(R,'UnsafeViewBackward0')}else if(X.shape.length===2&&b)rename(R,'AddmmBackward0');return R}
    case'Sequential':{let h=x;for(const c of m.children)h=callValue(c[1],[h],{},ctx);return h}
    case'Flatten':{const X=asTensor(x);const s=m.cfg.start_dim;return reshapeT(X,[...X.shape.slice(0,s),-1])}
    case'ReLU':return unary(x,'relu');case'Sigmoid':return unary(x,'sigmoid');case'Tanh':return unary(x,'tanh');case'Identity':return x;
    case'Softmax':return softmaxT(x,m.cfg.dim);case'LogSoftmax':return softmaxT(x,m.cfg.dim,true);
    case'Dropout':{if(!m.training||m.cfg.p===0)return x;const p=m.cfg.p,X=asTensor(x);if(p===1)return binop('*',X,I(0));const mask=Float64Array.from(X.data,()=>rand()>p?1/(1-p):0);return rename(binop('*',X,mkT(X.shape.slice(),mask,X.dtype)),'MulBackward0')}
    case'MSELoss':return mseLoss(args[0],args[1],m.cfg.reduction);case'L1Loss':return reduceLoss(unary(binop('-',args[0],args[1]),'abs'),m.cfg.reduction,'L1LossBackward0');
    case'HuberLoss':return huberLoss(args[0],args[1],m.cfg.reduction,m.cfg.delta);case'SmoothL1Loss':return smoothL1(args[0],args[1],m.cfg.reduction,m.cfg.beta);
    case'CrossEntropyLoss':return crossEntropy(args[0],args[1],m.cfg.reduction);case'BCELoss':return bceLoss(args[0],args[1],m.cfg.reduction,false);case'BCEWithLogitsLoss':return bceLoss(args[0],args[1],m.cfg.reduction,true)}
  err('NotImplementedError',`本页还没有实现 ${m.kind} 的前向计算`)}
const LOSSES=new Set(['MSELoss','L1Loss','HuberLoss','SmoothL1Loss','CrossEntropyLoss','BCELoss','BCEWithLogitsLoss']);
function nnRepr(m,ind=''){const c=m.cfg;switch(m.kind){
  case'Linear':return`Linear(in_features=${m.params.weight.shape[1]}, out_features=${m.params.weight.shape[0]}, bias=${m.params.bias?'True':'False'})`;
  case'Flatten':return`Flatten(start_dim=${c.start_dim}, end_dim=-1)`;case'Softmax':case'LogSoftmax':return`${m.kind}(dim=${c.dim})`;case'Dropout':return`Dropout(p=${pyFloatRepr(c.p)}, inplace=False)`;
  case'HuberLoss':case'SmoothL1Loss':return`${m.kind}()`;
  case'Sequential':return modListRepr('Sequential',m.children,ind);
  case'RNN':case'LSTM':case'GRU':return`${m.kind}(${c.input_size}, ${c.hidden_size}${c.num_layers>1?', num_layers='+c.num_layers:''}${c.bias?'':', bias=False'}${c.batch_first?', batch_first=True':''})`;
  case'Embedding':return`Embedding(${m.params.weight.shape[0]}, ${m.params.weight.shape[1]})`}return`${m.kind}()`}
function modListRepr(name,children,ind){if(!children.length)return name+'()';return name+'(\n'+children.map(([k,v])=>`${ind}  (${k}): ${modRepr(v,ind+'  ')}`).join('\n')+`\n${ind})`}
function modRepr(v,ind=''){if(v.t==='nn')return nnRepr(v,ind);if(v.t==='obj'&&isModuleObj(v))return modListRepr(v.cls.name,objChildren(v),ind);return repr(v)}
function isModuleCls(c){while(c){if(c===ModuleCls)return true;c=c.base}return false}
const isModuleObj=o=>o.t==='obj'&&isModuleCls(o.cls);
const isMod=v=>v&&(v.t==='nn'||isModuleObj(v));
function objChildren(o){const r=Object.entries(o.attrs).filter(([k,v])=>isMod(v));const d=o.attrs._modules;if(d&&d.t==='dict')for(const[k,v]of d.m.values())if(isMod(v))r.push([str(k),v]);return r}
function modChildren(m){return m.t==='nn'?m.children:objChildren(m)}
function namedParams(m,prefix='',dedup=true){const out=[];if(m.t==='nn'){for(const[k,p]of Object.entries(m.params))if(p)out.push([prefix+k,p])}else for(const[k,v]of Object.entries(m.attrs))if(isT(v)&&v.isParam)out.push([prefix+k,v]);
  for(const[k,c]of modChildren(m))out.push(...namedParams(c,prefix+k+'.',false));if(!dedup)return out;const seen=new Set();return out.filter(([k,p])=>seen.has(p)?false:(seen.add(p),true))}
function modName(m){return m.t==='nn'?m.kind:m.cls.name}
function loadStateDict(m,d,strict){if(d.t!=='dict')err('TypeError',`Expected state_dict to be dict-like, got ${typeName(d)}.`);
  const own=namedParams(m,'',false),ownKeys=new Set(own.map(x=>x[0])),given=[...d.m.values()].map(([k,v])=>[str(k),v]);
  const missing=own.map(x=>x[0]).filter(k=>!d.m.has(numKey(S(k)))),unexpected=given.map(x=>x[0]).filter(k=>!ownKeys.has(k)),sizeErr=[];
  for(const[k,p]of own){const e=d.m.get(numKey(S(k)));if(!e)continue;const v=e[1];if(!isT(v))err('TypeError',`expected torch.Tensor or Tensor-like object from checkpoint but received ${typeName(v)}`);
    if(v.shape.join()!==p.shape.join()){sizeErr.push(`size mismatch for ${k}: copying a param with shape torch.Size([${v.shape.join(', ')}]) from checkpoint, the shape in current model is torch.Size([${p.shape.join(', ')}]).`);continue}
    for(let q=0;q<p.data.length;q++)p.data[q]=castVal(v.data[q],p.dtype)}
  const msgs=[];if(strict){if(missing.length)msgs.push('Missing key(s) in state_dict: '+missing.map(k=>`"${k}"`).join(', ')+'. ');if(unexpected.length)msgs.splice(0,0,...[]),msgs.push('Unexpected key(s) in state_dict: '+unexpected.map(k=>`"${k}"`).join(', ')+'. ')}
  msgs.push(...sizeErr);if(msgs.length)err('RuntimeError',`Error(s) in loading state_dict for ${modName(m)}:\n\t`+msgs.join('\n\t'));
  return{t:'inckeys',missing,unexpected}}
function setTraining(m,v){if(m.t==='nn')m.training=v;else m.attrs.training=B(v);for(const[,c]of modChildren(m))setTraining(c,v)}
function moduleMethods(m){return{
  parameters:()=>({t:'iter',items:namedParams(m).map(x=>x[1]),pos:0}),named_parameters:()=>({t:'iter',items:namedParams(m).map(([k,p])=>tup([S(k),p])),pos:0}),
  children:()=>({t:'iter',items:modChildren(m).map(x=>x[1]),pos:0}),modules:()=>{const all=[m];const walk=x=>{for(const[,c]of modChildren(x)){all.push(c);walk(c)}};walk(m);return{t:'iter',items:all,pos:0}},
  train:a=>{setTraining(m,a.length?truthy(a[0]):true);return m},eval:()=>{setTraining(m,false);return m},
  apply:(a,kw,ctx)=>{const walk=x=>{for(const[,c]of modChildren(x))walk(c);callValue(a[0],[x],{},ctx)};walk(m);return m},
  zero_grad:()=>{for(const[,p]of namedParams(m))p.grad=null;return NONE},
  state_dict:()=>{const d=mkDict(namedParams(m,'',false).map(([k,p])=>{const t=mkT(p.shape.slice(),[],p.dtype);t.data=p.data;return[S(k),t]}));d.odict=true;return d},
  load_state_dict:(a,kw)=>loadStateDict(m,a[0],kw.strict?truthy(kw.strict):a[1]?truthy(a[1]):true),
  named_children:()=>({t:'iter',items:modChildren(m).map(([k,c])=>tup([S(k),c])),pos:0}),
  named_modules:()=>{const all=[tup([S(''),m])];const walk=(x,pre)=>{for(const[k,c]of modChildren(x)){all.push(tup([S(pre+k),c]));walk(c,pre+k+'.')}};walk(m,'');return{t:'iter',items:all,pos:0}},
  to:()=>m,cpu:()=>m,cuda:()=>m,double:()=>{for(const[,p]of namedParams(m)){p.dtype='float64'}return m},
  forward:(a,kw,ctx)=>nnForward(m,a,ctx),add_module:a=>{if(m.t==='nn')m.children.push([str(a[0]),a[1]]);else m.attrs[str(a[0])]=a[1];return NONE}}}
function nnAttr(m,name){const c=m.cfg;
  if(name in m.params){const p=m.params[name];return p||NONE}
  if(m.kind==='Linear'){if(name==='in_features')return I(m.params.weight.shape[1]);if(name==='out_features')return I(m.params.weight.shape[0])}
  if(c.attrs&&name in c.attrs)return c.attrs[name];
  if(name==='training')return B(m.training);if(name==='p'&&m.kind==='Dropout')return F(c.p);if(name==='reduction'&&LOSSES.has(m.kind))return S(c.reduction);
  const M=moduleMethods(m);if(M[name])return fn(name,M[name]);err('AttributeError',`'${m.kind}' object has no attribute '${name}'`)}
function nnMake(kind,a,kw){const bkw=(k,d)=>kw[k]!==undefined?truthy(kw[k]):d;
  switch(kind){case'RNN':case'LSTM':case'GRU':return rnnMake(kind,a,kw);
    case'Embedding':{const N=intOf(kwOr(kw,a,'num_embeddings',0)),D=intOf(kwOr(kw,a,'embedding_dim',1));return nnObj('Embedding',{attrs:{num_embeddings:I(N),embedding_dim:I(D)}},{weight:mkParam([N,D],Array.from({length:N*D},gauss))})}
    case'Linear':{const inF=intOf(kwOr(kw,a,'in_features',0)),outF=intOf(kwOr(kw,a,'out_features',1));return nnObj('Linear',{},linearInit(inF,outF,bkw('bias',a[2]?truthy(a[2]):true)))}
    case'Sequential':{if(a.length===1&&a[0].t==='dict')return nnObj('Sequential',{},{},[...a[0].m.values()].map(([k,v])=>[str(k),v]));return nnObj('Sequential',{},{},a.map((v,i)=>{if(!isMod(v))err('TypeError',`${typeName(v)} is not a Module subclass`);return[String(i),v]}))}
    case'Flatten':return nnObj('Flatten',{start_dim:kw.start_dim?intOf(kw.start_dim):a[0]?intOf(a[0]):1});
    case'Softmax':case'LogSoftmax':return nnObj(kind,{dim:kw.dim?intOf(kw.dim):a[0]?intOf(a[0]):null});
    case'Dropout':return nnObj('Dropout',{p:numOf(kwOr(kw,a,'p',0,F(0.5)))});
    case'HuberLoss':return nnObj(kind,{reduction:redOf(kw),delta:kw.delta?numOf(kw.delta):1});case'SmoothL1Loss':return nnObj(kind,{reduction:redOf(kw),beta:kw.beta?numOf(kw.beta):1});
    default:return nnObj(kind,{reduction:redOf(kw)})}}
// nn.Module：给 class MyNet(nn.Module) 继承用
const ModuleCls={t:'cls',name:'Module',base:null,attrs:Object.create(null)};
ModuleCls.attrs.__init__=fn('__init__',a=>{const self=a[0];self.attrs.training=B(true);const d=mkDict([]);d.odict=true;self.attrs._modules=d;return NONE});ModuleCls.attrs.__init__.method=true;
ModuleCls.attrs.__call__=fn('__call__',(a,kw,ctx)=>{const self=a[0];const f=classLookup(self.cls,'forward');if(!f)err('NotImplementedError',`Module [${self.cls.name}] is missing the required "forward" function`);return callValue({t:'bound',f,self},a.slice(1),kw,ctx)});ModuleCls.attrs.__call__.method=true;
for(const k of['parameters','named_parameters','children','named_children','modules','named_modules','train','eval','apply','zero_grad','state_dict','load_state_dict','to','cpu','cuda','add_module']){ModuleCls.attrs[k]=fn(k,(a,kw,ctx)=>moduleMethods(a[0])[k](a.slice(1),kw,ctx));ModuleCls.attrs[k].method=true}
const NN_KINDS=['RNN','LSTM','GRU','Embedding','Linear','Sequential','Flatten','ReLU','Sigmoid','Tanh','Softmax','LogSoftmax','Dropout','Identity','MSELoss','L1Loss','HuberLoss','SmoothL1Loss','CrossEntropyLoss','BCELoss','BCEWithLogitsLoss'];
const nnInit=mod('torch.nn.init',{
  normal_:fn('normal_',(a,kw)=>{const T=a[0],m=numOf(kwOr(kw,a,'mean',1,F(0))),s=numOf(kwOr(kw,a,'std',2,F(1)));for(let i=0;i<T.data.length;i++)T.data[i]=castVal(m+s*gauss(),T.dtype);return T}),
  uniform_:fn('uniform_',(a,kw)=>{const T=a[0],lo=numOf(kwOr(kw,a,'a',1,F(0))),hi=numOf(kwOr(kw,a,'b',2,F(1)));for(let i=0;i<T.data.length;i++)T.data[i]=castVal(lo+(hi-lo)*rand(),T.dtype);return T}),
  zeros_:fn('zeros_',a=>{a[0].data.fill(0);return a[0]}),ones_:fn('ones_',a=>{a[0].data.fill(1);return a[0]}),constant_:fn('constant_',a=>{a[0].data.fill(castVal(numOf(a[1]),a[0].dtype));return a[0]}),
  xavier_uniform_:fn('xavier_uniform_',(a,kw)=>{const T=a[0],[o,i]=T.shape,g=kw.gain?numOf(kw.gain):1,b=g*Math.sqrt(6/(i+o));for(let k=0;k<T.data.length;k++)T.data[k]=castVal((rand()*2-1)*b,T.dtype);return T}),
  xavier_normal_:fn('xavier_normal_',a=>{const T=a[0],[o,i]=T.shape,s=Math.sqrt(2/(i+o));for(let k=0;k<T.data.length;k++)T.data[k]=castVal(s*gauss(),T.dtype);return T})});
const nnF=mod('torch.nn.functional',{one_hot:fn('one_hot',(a,kw)=>{const T=asTensor(a[0]);if(T.dtype!=='int64')err('RuntimeError','one_hot is only applicable to index tensor of type LongTensor.');let C=intOf(kwOr(kw,a,'num_classes',1,I(-1)));if(C<0)C=Math.max(-1,...T.data)+1;const out=new Float64Array(T.data.length*C);T.data.forEach((v,i)=>{if(v<0||v>=C)err('RuntimeError','Class values must be smaller than num_classes.');out[i*C+v]=1});return mkT([...T.shape,C],out,'int64')}),relu:fn('relu',a=>unary(a[0],'relu')),sigmoid:fn('sigmoid',a=>unary(a[0],'sigmoid')),tanh:fn('tanh',a=>unary(a[0],'tanh')),
  softmax:fn('softmax',(a,kw)=>softmaxT(a[0],intOf(kwOr(kw,a,'dim',1)))),log_softmax:fn('log_softmax',(a,kw)=>softmaxT(a[0],intOf(kwOr(kw,a,'dim',1)),true)),
  cross_entropy:fn('cross_entropy',(a,kw)=>crossEntropy(a[0],a[1],redOf(kw))),mse_loss:fn('mse_loss',(a,kw)=>mseLoss(a[0],a[1],redOf(kw))),
  binary_cross_entropy:fn('binary_cross_entropy',(a,kw)=>bceLoss(a[0],a[1],redOf(kw),false)),binary_cross_entropy_with_logits:fn('binary_cross_entropy_with_logits',(a,kw)=>bceLoss(a[0],a[1],redOf(kw),true)),
  huber_loss:fn('huber_loss',(a,kw)=>huberLoss(a[0],a[1],redOf(kw),kw.delta?numOf(kw.delta):1)),dropout:fn('dropout',(a,kw)=>{const p=numOf(kwOr(kw,a,'p',1,F(0.5)));const tr=kw.training?truthy(kw.training):true;return tr?nnForward(nnObj('Dropout',{p}),[a[0]],null):a[0]})});
const nnUtils=mod('torch.nn.utils',{clip_grad_norm_:fn('clip_grad_norm_',(a,kw)=>{const ps=(isT(a[0])?[a[0]]:iterate(a[0])).filter(p=>p.grad),mx=numOf(kwOr(kw,a,'max_norm',1));let s=0;for(const p of ps)for(const v of p.grad.data)s+=v*v;const n=Math.fround(Math.sqrt(s)),k=mx/(n+1e-6);
    if(k<1)for(const p of ps)for(let i=0;i<p.grad.data.length;i++)p.grad.data[i]=castVal(p.grad.data[i]*k,p.grad.dtype);return mkT([],[n],'float32')}),
  clip_grad_value_:fn('clip_grad_value_',(a,kw)=>{const ps=(isT(a[0])?[a[0]]:iterate(a[0])).filter(p=>p.grad),c=numOf(kwOr(kw,a,'clip_value',1));for(const p of ps)for(let i=0;i<p.grad.data.length;i++)p.grad.data[i]=Math.max(-c,Math.min(c,p.grad.data[i]));return NONE})});
const nnMod=mod('torch.nn',{utils:nnUtils,Module:ModuleCls,init:nnInit,functional:nnF,Parameter:fn('Parameter',(a,kw)=>{const T=asTensor(a[0]);const p=mkParam(T.shape.slice(),T.data);p.dtype=T.dtype;if(kw.requires_grad&&!truthy(kw.requires_grad))p.requires_grad=false;return p})});
for(const k of NN_KINDS){nnMod.attrs[k]=fn(k,(a,kw)=>nnMake(k,a,kw));nnMod.attrs[k].nnKind=k}
// 优化器
function optMake(kind,a,kw){const items=a[0].t==='list'||a[0].t==='tuple'||a[0].t==='iter'?iterate(a[0]):[a[0]];const toList=v=>isT(v)?[v]:iterate(v).filter(isT);
  const defLr=numOf(kwOr(kw,a,'lr',1,F(kind==='SGD'?NaN:0.001)));if(kind==='SGD'&&Number.isNaN(defLr))err('TypeError',"SGD.__init__() missing 1 required positional argument: 'lr'");
  const raw={lr:kwOr(kw,a,'lr',1,F(defLr)),momentum:kw.momentum||I(0),weight_decay:kw.weight_decay||I(0)};const base={lr:defLr,momentum:numOf(raw.momentum),weight_decay:numOf(raw.weight_decay)};if(kind==='Adam'){const b=kw.betas?iterate(kw.betas).map(numOf):[0.9,0.999];base.b1=b[0];base.b2=b[1];base.eps=kw.eps?numOf(kw.eps):1e-8}
  const groups=items.length&&items[0].t==='dict'?items.map(d=>{const g={...base},r={...raw};let ps=[];for(const[k,v]of d.m.values()){const key=str(k);if(key==='params')ps=toList(v);else{g[key]=numOf(v);r[key]=v}}return{params:ps,g,r}}):[{params:items.filter(isT),g:base,r:raw}];
  if(!groups.some(x=>x.params.length))err('ValueError','optimizer got an empty parameter list');
  for(const G of groups)G.pg=mkDict([[S('params'),list(G.params)],[S('lr'),G.r.lr],[S('momentum'),G.r.momentum],[S('weight_decay'),G.r.weight_decay]]);
  return{t:'optim',kind,groups,params:groups.flatMap(G=>G.params),g:groups[0].g,pg:groups[0].pg,state:new Map(),t_:0}}
function optAttr(o,name){
  if(name==='param_groups')return list(o.groups.map(G=>G.pg));
  if(name==='zero_grad')return fn('zero_grad',(a,kw)=>{const none=kw.set_to_none?truthy(kw.set_to_none):true;for(const p of o.params){if(none)p.grad=null;else if(p.grad)p.grad.data.fill(0)}return NONE});
  if(name==='step')return fn('step',()=>{o.t_++;for(const G of o.groups){const g=G.g,L=numOf(G.pg.m.get(numKey(S('lr')))[1]),wd=numOf(G.pg.m.get(numKey(S('weight_decay')))[1]);for(const p of G.params){if(!p.grad)continue;const gr=Float64Array.from(p.grad.data);if(wd)for(let i=0;i<gr.length;i++)gr[i]+=wd*p.data[i];
      if(o.kind==='SGD'){let d=gr;if(g.momentum){let v=o.state.get(p);if(!v){v=Float64Array.from(gr)}else for(let i=0;i<v.length;i++)v[i]=g.momentum*v[i]+gr[i];o.state.set(p,v);d=v}for(let i=0;i<p.data.length;i++)p.data[i]=castVal(p.data[i]-L*d[i],p.dtype)}
      else{let s=o.state.get(p);if(!s){s={m:new Float64Array(gr.length),v:new Float64Array(gr.length)};o.state.set(p,s)}const c1=1-g.b1**o.t_,c2=1-g.b2**o.t_;for(let i=0;i<gr.length;i++){s.m[i]=g.b1*s.m[i]+(1-g.b1)*gr[i];s.v[i]=g.b2*s.v[i]+(1-g.b2)*gr[i]*gr[i];p.data[i]=castVal(p.data[i]-L*(s.m[i]/c1)/(Math.sqrt(s.v[i]/c2)+g.eps),p.dtype)}}}}return NONE});
  err('AttributeError',`'${o.kind}' object has no attribute '${name}'`)}
function optRepr(o){const g=o.g,lines=o.kind==='SGD'?[['dampening','0'],['differentiable','False'],['foreach','None'],['fused','None'],['lr',pyFloatRepr(g.lr)],['maximize','False'],['momentum',g.momentum?pyFloatRepr(g.momentum):'0'],['nesterov','False'],['weight_decay',g.weight_decay?pyFloatRepr(g.weight_decay):'0']]
  :[['amsgrad','False'],['betas',`(${pyFloatRepr(g.b1)}, ${pyFloatRepr(g.b2)})`],['capturable','False'],['decoupled_weight_decay','False'],['differentiable','False'],['eps',String(g.eps)],['foreach','None'],['fused','None'],['lr',pyFloatRepr(g.lr)],['maximize','False'],['weight_decay',g.weight_decay?pyFloatRepr(g.weight_decay):'0']];
  return`${o.kind} (\nParameter Group 0\n${lines.map(([k,v])=>`    ${k}: ${v}`).join('\n')}\n)`}
const OptimizerCls={t:'cls',name:'Optimizer',base:null,attrs:Object.create(null)};
const optimMod=mod('torch.optim',{Optimizer:OptimizerCls,SGD:fn('SGD',(a,kw)=>optMake('SGD',a,kw)),Adam:fn('Adam',(a,kw)=>optMake('Adam',a,kw))});
// 数据：TensorDataset 和 DataLoader
const dataMod=mod('torch.utils.data',{TensorDataset:fn('TensorDataset',a=>{const ts=a.map(asTensor);if(ts.some(t=>t.shape[0]!==ts[0].shape[0]))err('AssertionError','Size mismatch between tensors');return{t:'tdataset',tensors:ts}}),
  DataLoader:fn('DataLoader',(a,kw)=>({t:'loader',ds:a[0],bs:intOf(kwOr(kw,a,'batch_size',1,I(1))),shuffle:truthy(kwOr(kw,a,'shuffle',2,B(false))),drop_last:kw.drop_last?truthy(kw.drop_last):false}))});
function dsLen(ds,ctx){if(ds.t==='tdataset')return ds.tensors[0].shape[0];if(ds.t==='fmnist')return ds.n;if(ds.t==='obj'){const f=classLookup(ds.cls,'__len__');if(f)return intOf(callValue({t:'bound',f,self:ds},[],{},ctx))}return pyLen(ds)}
function* loaderGen(L,ctx){const n=dsLen(L.ds,ctx),idx=[...Array(n).keys()];if(L.shuffle)for(let i=n-1;i>0;i--){const j=Math.floor(rand()*(i+1));[idx[i],idx[j]]=[idx[j],idx[i]]}
  for(let s=0;s<n;s+=L.bs){const b=idx.slice(s,s+L.bs);if(L.drop_last&&b.length<L.bs)break;
    if(L.ds.t==='fmnist'){yield fmBatch(L.ds,b);continue}
    if(L.ds.t==='tdataset'){const bi=mkT([b.length],b,'int64');yield list(L.ds.tensors.map(t=>{const r=getItem(t,[{tensor:bi}]);r.meta=null;return r}));continue}
    const items=b.map(i=>getSub(L.ds,{items:[{e:{k:'const',v:I(i)}}]},null,ctx));const first=items[0];const parts=first.items?first.items.map((_,k)=>items.map(it=>it.items[k])):[items];
    yield list(parts.map(col=>isT(col[0])?stack([list(col)],{dim:I(0)}):mkT([col.length],col.map(numOf),col.every(v=>v.t==='int')?'int64':'float32')))}}
function loaderBatches(L,ctx){return[...loaderGen(L,ctx)]}

torch.attrs.nn=nnMod;torch.attrs.optim=optimMod;torch.attrs.utils=mod('torch.utils',{data:dataMod});torch.attrs.softmax=fn('softmax',(a,kw)=>softmaxT(a[0],intOf(kwOr(kw,a,'dim',1))));torch.attrs.log_softmax=fn('log_softmax',(a,kw)=>softmaxT(a[0],intOf(kwOr(kw,a,'dim',1)),true));
