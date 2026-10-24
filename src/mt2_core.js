// ============ 迷你 PyTorch：在浏览器里运行书上的代码 ============
// 实现课程用到的 Python 子集（缩进代码块、函数、推导式、f-string）、torch（含自动求导）、
// 一个小型 pandas、文件读写和 d2l 画图。打印格式对齐 PyTorch / pandas 的 repr 规则。
const MT=(function(){
class PyErr extends Error{constructor(type,msg){super(msg);this.pyType=type}}
const err=(t,m)=>{throw new PyErr(t,m)};
let idCounter=0;const newId=()=>4437121520+(idCounter++)*48;
let seed=20251023;
function rand(){seed|=0;seed=seed+0x6D2B79F5|0;let t=Math.imul(seed^seed>>>15,1|seed);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}
function gauss(){let u=0;while(!u)u=rand();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*rand())}

const RANK={bool:0,int64:1,float32:2,float64:3};
const isFloatDt=d=>d==='float32'||d==='float64';
function castVal(v,dt){return dt==='float32'?Math.fround(v):dt==='bool'?(v?1:0):dt==='int64'?Math.trunc(v):v}
function convData(data,dtype){const n=data.length,out=new Float64Array(n);
  if(dtype==='float32'){if(data instanceof Float32Array)out.set(data);else for(let i=0;i<n;i++)out[i]=Math.fround(data[i])}
  else if(dtype==='float64'){if(ArrayBuffer.isView(data))out.set(data);else for(let i=0;i<n;i++)out[i]=+data[i]}
  else if(dtype==='int64'){for(let i=0;i<n;i++)out[i]=Math.trunc(data[i])}else{for(let i=0;i<n;i++)out[i]=data[i]?1:0}return out}
class Tensor{constructor(shape,data,dtype){this.shape=shape;this.dtype=dtype;this.data=convData(data,dtype);this.id=newId();this.meta=null;this.requires_grad=false;this.grad=null;this.grad_fn=null;this.np=false}}
const isT=v=>v instanceof Tensor;
const prod=s=>s.reduce((a,b)=>a*b,1);
function strides(s){const st=new Array(s.length);let k=1;for(let i=s.length-1;i>=0;i--){st[i]=k;k*=s[i]}return st}
const I=v=>({t:'int',v}),F=v=>({t:'float',v}),B=v=>({t:'bool',v:!!v}),S=v=>({t:'str',v}),NONE={t:'none'};
const tup=items=>({t:'tuple',items}),list=items=>({t:'list',items});
const DUMMY={t:'dummy'};
const snapshot=T=>{const c=new Tensor(T.shape.slice(),T.data,T.dtype);c.id=T.id;c.np=T.np;return c};
const mkT=(shape,data,dtype,np)=>{const t=new Tensor(shape,data,dtype);t.np=!!np;return t};

// ---------- 自动求导 ----------
let gradEnabled=true;
function track(out,inputs,name,back){if(!gradEnabled||!inputs.some(t=>isT(t)&&t.requires_grad))return out;out.requires_grad=true;out.grad_fn={name,inputs,back,id:newId()};return out}
function unb(g,outShape,inShape){ // 把广播后的梯度加回到原来的形状
  if(outShape.join()===inShape.join())return g;const n=prod(inShape),res=new Float64Array(n),k=outShape.length-inShape.length,ist=strides(inShape),idx=new Array(outShape.length).fill(0);
  for(let f=0;f<g.length;f++){let off=0;for(let d=k;d<outShape.length;d++)if(inShape[d-k]!==1)off+=idx[d]*ist[d-k];res[off]+=g[f];for(let d=outShape.length-1;d>=0;d--){if(++idx[d]<outShape[d])break;idx[d]=0}}return res}
function backward(root,gradient,retain){
  if(!root.requires_grad)err('RuntimeError','element 0 of tensors does not require grad and does not have a grad_fn');
  let g;if(!gradient||gradient.t==='none'){if(root.data.length!==1)err('RuntimeError','grad can be implicitly created only for scalar outputs');g=new Float64Array([1])}else g=Float64Array.from(asTensor(gradient).data);
  const order=[],seen=new Set();
  const visit=t=>{if(seen.has(t))return;seen.add(t);if(t.grad_fn){if(t.grad_fn.freed)err('RuntimeError','Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed). Saved intermediate values of the graph are freed when you call .backward() or autograd.grad(). Specify retain_graph=True if you need to backward through the graph a second time or if you need to access saved tensors after calling backward.');
    for(const i of t.grad_fn.inputs)if(isT(i)&&i.requires_grad)visit(i)}order.push(t)};visit(root);
  const grads=new Map([[root,g]]);
  for(let k=order.length-1;k>=0;k--){const t=order[k],gt=grads.get(t);if(!gt)continue;
    if(t.grad_fn){const gi=t.grad_fn.back(gt);t.grad_fn.inputs.forEach((inp,j)=>{if(isT(inp)&&inp.requires_grad&&gi[j]){const prev=grads.get(inp);if(prev){for(let q=0;q<prev.length;q++)prev[q]+=gi[j][q]}else grads.set(inp,Float64Array.from(gi[j]))}})}
    else{if(!t.grad){t.grad=new Tensor(t.shape.slice(),gt,t.dtype)}else{for(let q=0;q<gt.length;q++)t.grad.data[q]=castVal(t.grad.data[q]+gt[q],t.dtype)}}}
  if(!retain)for(const t of order)if(t.grad_fn)t.grad_fn.freed=true;return NONE}
const checkInplace=T=>{if(gradEnabled&&T.requires_grad&&!T.grad_fn)err('RuntimeError','a leaf Variable that requires grad is being used in an in-place operation.')};
