
// ---------- 数值打印（对齐 torch 的 _tensor_str） ----------
function sciStr(v){return v.toExponential(4).replace(/e([+-])(\d)$/,'e$10$2')}
function elemFormatter(vals,dtype){
  if(dtype==='bool')return v=>v?'True':'False';
  if(dtype==='int64')return v=>String(v);
  const fin=vals.filter(Number.isFinite),nz=fin.map(Math.abs).filter(v=>v>0);
  const intMode=fin.every(v=>Math.ceil(v)===v);
  let sci=false;
  if(nz.length){const mx=Math.max(...nz),mn=Math.min(...nz);sci=mx/mn>1000||mx>1e8||mn<1e-4}
  return v=>{if(Number.isNaN(v))return'nan';if(!Number.isFinite(v))return v>0?'inf':'-inf';
    if(sci)return sciStr(v);if(intMode)return(Object.is(v,-0)?'-0':v.toFixed(0))+'.';return v.toFixed(4)}}
function summarizedIdx(n,summ){return summ&&n>6?[0,1,2,-1,n-3,n-2,n-1]:[...Array(n).keys()]}
function nestedStr(T,indent0,f,w){const nd=T.shape.length,st=strides(T.shape),summ=T.data.length>1000,pad=v=>f(v).padStart(w);
  const rec=(dim,off,indent)=>{const n=T.shape[dim],idx=summarizedIdx(n,summ);
    if(dim===nd-1){const items=idx.map(i=>i<0?'...':pad(T.data[off+i*st[dim]]));const per=Math.max(1,Math.floor((80-indent)/(w+2)));const lines=[];for(let i=0;i<items.length;i+=per)lines.push(items.slice(i,i+per).join(', '));return'['+lines.join(',\n'+' '.repeat(indent+1))+']'}
    const subs=idx.map(i=>i<0?'...':rec(dim+1,off+i*st[dim],indent+1));return'['+subs.join(','+'\n'.repeat(nd-dim-1)+' '.repeat(indent+1))+']'};
  return rec(0,0,indent0)}
function shownVals(T){const nd=T.shape.length,st=strides(T.shape),summ=T.data.length>1000,out=[];const collect=(dim,off)=>{if(dim===nd){out.push(T.data[off]);return}for(const i of summarizedIdx(T.shape[dim],summ))if(i>=0)collect(dim+1,off+i*st[dim])};collect(0,0);return out}
function tensorRepr(T){if(T.np)return npRepr(T);
  const shown=shownVals(T),f=elemFormatter(shown,T.dtype),w=isFloatDt(T.dtype)?Math.max(1,...shown.filter(v=>Number.isFinite(v)&&v!==0).map(v=>f(v).length)):Math.max(1,...shown.map(v=>f(v).length));
  let body=T.shape.length===0?f(T.data[0]):nestedStr(T,7,f,w);if(T.data.length===0)body='[]';
  const sufs=[];if(!(T.dtype==='int64'||T.dtype==='float32'||T.dtype==='bool'))sufs.push(`dtype=torch.${T.dtype}`);
  if(T.grad_fn)sufs.push(`grad_fn=<${T.grad_fn.name}>`);else if(T.requires_grad)sufs.push('requires_grad=True');
  // 对齐 torch._tensor_str._add_suffixes：放不下时换行
  let s='tensor('+body,last=s.length-s.lastIndexOf('\n')+1;for(const x of sufs){if(last+x.length+2>80){s+=',\n'+' '.repeat(7)+x;last=7+x.length}else{s+=', '+x;last+=x.length+2}}
  return s+')'}
function npShort(v,f32){if(Number.isNaN(v))return'nan';if(!Number.isFinite(v))return v>0?'inf':'-inf';let s;if(f32){for(let p=1;p<=9;p++){s=v.toPrecision(p);if(Math.fround(+s)===v)break}s=String(+s)}else s=String(v);
  if(/e/.test(s)){const n=+s;s=Math.abs(n)<1?n.toFixed(Math.min(20,Math.ceil(-Math.log10(Math.abs(n)))+8)).replace(/0+$/,''):n.toFixed(0)}
  const k=s.indexOf('.');if(k>=0&&s.length-k-1>8){s=v.toFixed(8).replace(/0+$/,'');if(s.endsWith('.'))s=s}return s}
function npElemStrs(T){const vals=[...T.data];if(T.dtype==='bool')return vals.map(v=>v?' True':'False');if(T.dtype==='int64'){const w=Math.max(...vals.map(v=>String(v).length));return vals.map(v=>String(v).padStart(w))}
  const f32=T.dtype==='float32',parts=vals.map(v=>{const s=npShort(v,f32);if(!/^-?\d/.test(s))return[s,null];const k=s.indexOf('.');return k<0?[s,'']:[s.slice(0,k),s.slice(k+1)]});
  const L=Math.max(...parts.map(p=>p[0].length)),Fm=Math.max(0,...parts.map(p=>p[1]==null?0:p[1].length));return parts.map(([a,b])=>b==null?a.padStart(L+1+Fm):a.padStart(L)+'.'+b.padEnd(Fm))}
function npRepr(T,asStr){const strs=npElemStrs(T),st=strides(T.shape),nd=T.shape.length,sep=asStr?' ':', ';if(nd===0)return strs[0].trim();const w=Math.max(...strs.map(s=>s.length));
  const base=asStr?1:7;const rec=(dim,off,indent)=>{const n=T.shape[dim];if(dim===nd-1){const items=[];for(let k=0;k<n;k++)items.push(strs[off+k*st[dim]]);const per=Math.max(1,Math.floor((75-indent)/(w+sep.length)));const lines=[];for(let k=0;k<items.length;k+=per)lines.push(items.slice(k,k+per).join(sep));return'['+lines.join((asStr?'':',')+'\n'+' '.repeat(indent+1))+']'}
    const subs=[];for(let k=0;k<n;k++)subs.push(rec(dim+1,off+k*st[dim],indent+1));return'['+subs.join((asStr?'':',')+'\n'.repeat(nd-dim-1)+' '.repeat(indent+1))+']'};
  const body=rec(0,0,base-1);return asStr?body:'array('+body+(T.dtype==='float32'?', dtype=float32':'')+')'}
function pyFloatRepr(v){if(Number.isNaN(v))return'nan';if(!Number.isFinite(v))return v>0?'inf':'-inf';if(v===0)return Object.is(v,-0)?'-0.0':'0.0';
  const e=v.toExponential(),m=e.match(/^(-?)(\d)(?:\.(\d+))?e([+-]\d+)$/),sign=m[1],digits=m[2]+(m[3]||''),exp=+m[4];
  if(exp<-4||exp>=16){const mant=digits.length>1?digits[0]+'.'+digits.slice(1):digits;return sign+mant+'e'+(exp<0?'-':'+')+String(Math.abs(exp)).padStart(2,'0')}
  let s;if(exp>=0){s=digits.length>exp+1?digits.slice(0,exp+1)+'.'+digits.slice(exp+1):digits+'0'.repeat(exp+1-digits.length)+'.0'}else s='0.'+'0'.repeat(-exp-1)+digits;return sign+s}
function strRepr(s){const q=s.includes("'")&&!s.includes('"')?'"':"'";return q+s.replace(/\\/g,'\\\\').replace(/\n/g,'\\n').replace(/\t/g,'\\t')+q}
function repr(v){if(isT(v))return(v.isParam?'Parameter containing:\n':'')+tensorRepr(v);
  switch(v.t){case'int':return String(v.v);case'float':return v.np?`np.float64(${pyFloatRepr(v.v)})`:pyFloatRepr(v.v);case'bool':return v.v?'True':'False';case'none':return'None';case'str':return strRepr(v.v);
    case'tuple':return'('+v.items.map(repr).join(', ')+(v.items.length===1?',':'')+')';case'list':if(v.isSet)return v.items.length?'{'+v.items.map(repr).join(', ')+'}':'set()';return(v.view?v.view+'(':'')+'['+v.items.map(repr).join(', ')+']'+(v.view?')':'');case'inckeys':return v.missing.length||v.unexpected.length?`_IncompatibleKeys(missing_keys=[${v.missing.map(strRepr).join(', ')}], unexpected_keys=[${v.unexpected.map(strRepr).join(', ')}])`:'<All keys matched successfully>';
    case'dict':{const body='{'+[...v.m.values()].map(([k,x])=>repr(k)+': '+repr(x)).join(', ')+'}';if(v.counter){const r=[...v.m.values()].sort((x,y)=>y[1].v-x[1].v);return r.length?'Counter({'+r.map(([k,x])=>repr(k)+': '+repr(x)).join(', ')+'})':'Counter()'}if(v.dflt)return`defaultdict(${repr(v.dflt).replace(/^<built-in function (\w+)>$/,"<class '$1'>")}, ${body})`;return v.odict?(v.m.size?`OrderedDict(${body})`:'OrderedDict()'):body}
    case'range':return`range(${v.start}, ${v.stop}${v.step!==1?', '+v.step:''})`;
    case'size':return`torch.Size([${v.dims.join(', ')}])`;case'dtype':return`torch.${v.name}`;case'type':return`<class '${v.name}'>`;case'mod':return`<module '${v.name}'>`;
    case'fn':return`<built-in function ${v.name}>`;case'func':return`<function ${v.name} at 0x${(v.id||4400000000).toString(16)}>`;case'gradfn':return`<${v.g.name} object at 0x${v.g.id.toString(16)}>`;
    case'df':return dfRepr(v);case'series':return seriesRepr(v);case'index':return`Index([${v.items.map(x=>strRepr(x)).join(', ')}], dtype='str')`;
    case'iter':return'<generator object>';case'nn':return nnRepr(v);case'fmnist':return fmRepr(v);case'transform':return tfRepr(v);case'fig':return`<Figure size ${v.cols*150}x${v.rows*150} with ${v.axes.length} Axes>`;case'ax':return'<Axes: >';case'axarr':return`array([${v.fig.axes.map(()=>'<Axes: >').join(', ')}], dtype=object)`;case'pil':return`<PIL.Image.Image image mode=L size=${v.W}x${v.H}>`;case'obj':return isModuleObj(v)?modRepr(v):`<__main__.${v.cls.name} object at 0x${(v.id||(v.id=newId())).toString(16)}>`;case'cls':return`<class '__main__.${v.name}'>`;case'optim':return optRepr(v);case'loader':return`<torch.utils.data.dataloader.DataLoader object at 0x${(v.id||(v.id=newId())).toString(16)}>`;case'tdataset':return`<torch.utils.data.dataset.TensorDataset object at 0x${(v.id||(v.id=newId())).toString(16)}>`;case'bound':return`<bound method ${v.f.name} of ${repr(v.self)}>`;case'file':return`<_io.TextIOWrapper name='${v.path}' mode='${v.mode}'>`;case'dummy':return'<object>';case'dist':return`${v.name}()`}return String(v)}
const str=v=>v.t==='str'?v.v:(isT(v)&&v.np)?npRepr(v,true):repr(v);

// ---------- 类型与广播 ----------
function asTensor(v){if(isT(v))return v;if(v.t==='int'){const t=new Tensor([],[v.v],'int64');t.weak=true;return t}if(v.t==='float'){const t=new Tensor([],[v.v],'float64');t.weak=true;t.weakFloat=true;return t}if(v.t==='bool'){const t=new Tensor([],[v.v],'bool');t.weak=true;return t}
  if(v.t==='list'||v.t==='tuple'){const n=nested(v);return mkT(n.shape,n.flat,n.dt==='float32'?'float64':n.dt,true)}
  err('TypeError',`unsupported operand type: '${typeName(v)}'`)}
function resultDtype(A,B){
  if(A.weak&&!B.weak){if(isFloatDt(B.dtype))return B.dtype;if(A.weakFloat)return B.np?'float64':'float32';if(B.dtype==='int64')return'int64';return A.dtype==='bool'?'bool':'int64'}
  if(B.weak&&!A.weak)return resultDtype(B,A);
  if(A.weak&&B.weak){if(A.weakFloat||B.weakFloat)return'float32';return RANK[A.dtype]>=RANK[B.dtype]?A.dtype:B.dtype}
  return RANK[A.dtype]>=RANK[B.dtype]?A.dtype:B.dtype}
function bshape(a,b){const n=Math.max(a.length,b.length),o=[];for(let i=0;i<n;i++){const x=a[a.length-n+i]??1,y=b[b.length-n+i]??1;
  if(x!==y&&x!==1&&y!==1)err('RuntimeError',`The size of tensor a (${x}) must match the size of tensor b (${y}) at non-singleton dimension ${i}`);o.push(x===1?y:x)}return o}
function bidx(shape,outShape){const n=outShape.length,st=strides(shape),k=n-shape.length;return outShape.map((d,i)=>i<k||shape[i-k]===1?0:st[i-k])}
const OPS={'+':(a,b)=>a+b,'-':(a,b)=>a-b,'*':(a,b)=>a*b,'/':(a,b)=>a/b,'**':(a,b)=>a**b,'//':(a,b)=>Math.floor(a/b),'%':(a,b)=>a-Math.floor(a/b)*b,
  '==':(a,b)=>a===b,'!=':(a,b)=>a!==b,'<':(a,b)=>a<b,'>':(a,b)=>a>b,'<=':(a,b)=>a<=b,'>=':(a,b)=>a>=b,'&':(a,b)=>a&&b,'|':(a,b)=>a||b};
const CMP=new Set(['==','!=','<','>','<=','>=']);
const BNAME={'+':'AddBackward0','-':'SubBackward0','*':'MulBackward0','/':'DivBackward0','**':'PowBackward0'};
function pyNum(v){if(v.t==='int'||v.t==='float')return v.v;if(v.t==='bool')return v.v?1:0;return null}
function binop(op,x,y){
  const xT=isT(x)||x.t==='df'||x.t==='series',yT=isT(y)||y.t==='df'||y.t==='series';
  if(x.t==='series'||y.t==='series')return seriesBinop(op,x,y);
  if(!xT&&!yT){ // Python 值
    if((x.t==='list'||x.t==='tuple')&&(y.t===x.t)){if(op==='+')return{t:x.t,items:[...x.items,...y.items]};if(op==='=='||op==='!=')return B((op==='==')===pyEq(x,y))}
    if((x.t==='list'||x.t==='tuple'||x.t==='str')&&op==='*'&&y.t==='int'){if(x.t==='str')return S(x.v.repeat(Math.max(0,y.v)));const r=[];for(let i=0;i<y.v;i++)r.push(...x.items);return{t:x.t,items:r}}
    if(x.t==='str'||y.t==='str'){if(op==='+'&&x.t==='str'&&y.t==='str')return S(x.v+y.v);if(op==='=='||op==='!=')return B((op==='==')===pyEq(x,y));if(CMP.has(op)&&x.t==='str'&&y.t==='str')return B(OPS[op](x.v,y.v));if(op==='%'&&x.t==='str')return S(percentFormat(x.v,y));err('TypeError',`unsupported operand type(s) for ${op}: '${typeName(x)}' and '${typeName(y)}'`)}
    if(op==='=='||op==='!='){const a=pyNum(x),b=pyNum(y);if(a==null||b==null)return B((op==='==')===pyEq(x,y))}
    const a=pyNum(x),b=pyNum(y);if(a==null||b==null)err('TypeError',`unsupported operand type(s) for ${op}: '${typeName(x)}' and '${typeName(y)}'`);
    if(CMP.has(op))return B(OPS[op](a,b));
    if((op==='/'||op==='//'||op==='%')&&b===0)err('ZeroDivisionError',op==='/'?'division by zero':'integer division or modulo by zero');
    if(op==='&'||op==='|'){const r=op==='&'?(a&b):(a|b);return x.t==='bool'&&y.t==='bool'?B(r):I(r)}
    const fl=x.t==='float'||y.t==='float'||op==='/'||(op==='**'&&b<0);const r=OPS[op](a,b);const res=fl?F(r):I(r);if((x.np||y.np)&&fl)res.np=true;return res}
  const A=asTensor(x),Bt=asTensor(y),out=bshape(A.shape,Bt.shape);let dt=resultDtype(A,Bt);const np=A.np||Bt.np;
  if(CMP.has(op))dt='bool';else if(op==='/'&&!isFloatDt(dt))dt=np?'float64':'float32';else if(op==='**'&&Bt.weakFloat&&!isFloatDt(dt))dt='float32';else if(dt==='bool'&&op!=='*'&&op!=='&'&&op!=='|')dt=op==='+'?'bool':'int64';
  const n=prod(out),sa=bidx(A.shape,out),sb=bidx(Bt.shape,out),res=new Float64Array(n),idx=new Array(out.length).fill(0),f=OPS[op];
  const needG=gradEnabled&&!CMP.has(op)&&(A.requires_grad||Bt.requires_grad);const ea=needG?new Float64Array(n):null,eb=needG?new Float64Array(n):null;
  for(let k=0;k<n;k++){let ia=0,ib=0;for(let d=0;d<out.length;d++){ia+=idx[d]*sa[d];ib+=idx[d]*sb[d]}const av=A.data[ia],bv=Bt.data[ib];let r=f(av,bv);if(needG){ea[k]=av;eb[k]=bv}
    if(dt==='bool')r=r?1:0;res[k]=+r;for(let d=out.length-1;d>=0;d--){if(++idx[d]<out[d])break;idx[d]=0}}
  if((op==='//'||op==='%')&&dt==='int64'&&Bt.data.some(v=>v===0))err('RuntimeError','ZeroDivisionError');
  const T=mkT(out,res,dt,np);
  if(isT(x)&&isT(y)&&!x.weak&&!y.weak&&(x.shape.join()!==y.shape.join())&&out.length<=3&&n<=300)T.meta={kind:'bcast',a:snapshot(A),b:snapshot(Bt),op};
  if(needG&&BNAME[op]){let name=BNAME[op];if(op==='**')name=A.weak?'PowBackward2':Bt.requires_grad?'PowBackward1':'PowBackward0';else if(op==='-'&&A.weak)name='RsubBackward1';else if(op==='/'&&A.weak)name='MulBackward0';const outV=T.data;
    track(T,[A,Bt],name,g=>{const ga=new Float64Array(n),gb=new Float64Array(n);for(let k=0;k<n;k++){const a=ea[k],b=eb[k];
      switch(op){case'+':ga[k]=g[k];gb[k]=g[k];break;case'-':ga[k]=g[k];gb[k]=-g[k];break;case'*':ga[k]=g[k]*b;gb[k]=g[k]*a;break;case'/':ga[k]=g[k]/b;gb[k]=-g[k]*a/(b*b);break;case'**':ga[k]=g[k]*b*a**(b-1);gb[k]=g[k]*outV[k]*Math.log(a);break}}
      return[unb(ga,out,A.shape),unb(gb,out,Bt.shape)]})}
  return T}
function pyEq(a,b){if(isT(a)||isT(b))return false;const na=pyNum(a),nb=pyNum(b);if(na!=null&&nb!=null)return na===nb;if(a.t!==b.t)return false;
  if(a.t==='str')return a.v===b.v;if(a.t==='none')return true;if(a.items)return a.items.length===b.items.length&&a.items.every((x,i)=>pyEq(x,b.items[i]));if(a.t==='size')return a.dims.join()===b.dims.join();if(a.t==='dtype')return a.name===b.name;return a===b}
function typeName(v){if(isT(v))return v.np?'numpy.ndarray':'Tensor';return{int:'int',float:'float',bool:'bool',str:'str',none:'NoneType',tuple:'tuple',list:'list',dict:'dict',range:'range',size:'Size',dtype:'dtype',fn:'builtin_function_or_method',func:'function',mod:'module',type:'type',df:'DataFrame',series:'Series',iter:'generator',file:'TextIOWrapper',optim:v.kind,loader:'DataLoader',tdataset:'TensorDataset'}[v.t]||(v.t==='nn'?v.kind:v.t==='obj'?v.cls.name:v.t)}

// ---------- 创建张量 ----------
function nested(v){if(isT(v))return{shape:v.shape.slice(),flat:[...v.data],dt:v.dtype,np:v.np};
  if(v.t==='list'||v.t==='tuple'){if(!v.items.length)return{shape:[0],flat:[],dt:'float32'};const subs=v.items.map(nested);const s0=subs[0].shape.join();
    if(subs.some(s=>s.shape.join()!==s0))err('ValueError',`expected sequence of length ${subs[0].shape[0]} at dim 1 (got ${subs.find(s=>s.shape.join()!==s0).shape[0]})`);
    let dt='bool';for(const s of subs)if(RANK[s.dt]>RANK[dt])dt=s.dt;return{shape:[v.items.length,...subs[0].shape],flat:subs.flatMap(s=>s.flat),dt}}
  if(v.t==='int')return{shape:[],flat:[v.v],dt:'int64'};if(v.t==='float')return{shape:[],flat:[v.v],dt:'float32'};if(v.t==='bool')return{shape:[],flat:[v.v?1:0],dt:'bool'};
  if(v.t==='objarr')err('TypeError',"can't convert np.ndarray of type numpy.object_. The only supported types are: float64, float32, float16, complex64, complex128, int64, int32, int16, int8, uint64, uint32, uint16, uint8, and bool.");
  err('TypeError',`Could not infer dtype of ${typeName(v)}`)}
function dimsArg(args){if(args.length===1&&(args[0].t==='tuple'||args[0].t==='list'||args[0].t==='size'))return(args[0].dims||args[0].items.map(intOf));return args.map(intOf)}
function intOf(v){if(v.t==='int')return v.v;if(v.t==='bool')return v.v?1:0;if(isT(v)&&v.data.length===1&&!isFloatDt(v.dtype))return v.data[0];err('TypeError',`期望整数，得到 ${typeName(v)}`)}
function numOf(v){const n=pyNum(v);if(n!=null)return n;if(isT(v)&&v.data.length===1)return v.data[0];err('TypeError',`期望数字，得到 ${typeName(v)}`)}
const dtypeKw=(kw,def)=>{if(!kw.dtype||kw.dtype.t==='none')return def;if(kw.dtype.t==='dtype')return kw.dtype.name;if(kw.dtype.t==='type'||(kw.dtype.t==='fn'&&kw.dtype.typeName)){return{float:'float64',int:'int64',bool:'bool'}[kw.dtype.typeName||kw.dtype.name]||def}err('TypeError','dtype 必须是 torch.float32 这样的类型')};
const withGrad=(T,kw)=>{if(kw.requires_grad&&kw.requires_grad.v){if(!isFloatDt(T.dtype))err('RuntimeError','Only Tensors of floating point and complex dtype can require gradients');T.requires_grad=true}return T};
function reshapeT(T,dims){let neg=dims.indexOf(-1);const known=prod(dims.filter(d=>d!==-1));
  if(neg>=0){if(known===0||T.data.length%known)err('RuntimeError',`shape '[${dims.join(', ')}]' is invalid for input of size ${T.data.length}`);dims=dims.slice();dims[neg]=T.data.length/known}
  if(prod(dims)!==T.data.length)err('RuntimeError',`shape '[${dims.join(', ')}]' is invalid for input of size ${T.data.length}`);
  const R=mkT(dims,[],T.dtype,T.np);R.data=T.data;return track(R,[T],'ViewBackward0',g=>[g])}
function normDims(T,dim){const nd=T.shape.length;const ds=(Array.isArray(dim)?dim:[dim]).map(d=>{if(d<0)d+=nd;if(d<0||d>=nd)err('IndexError',`Dimension out of range (expected to be in range of [${-nd}, ${nd-1}], but got ${d})`);return d});return[...new Set(ds)].sort((a,b)=>a-b)}
function clampT(T,lo,hi){const L=lo&&lo.t!=='none'?numOf(lo):-Infinity,H=hi&&hi.t!=='none'?numOf(hi):Infinity;if(L===-Infinity&&H===Infinity)err('RuntimeError',"torch.clamp: At least one of 'min' or 'max' must not be None");
  const dt=T.dtype==='bool'?'int64':T.dtype,out=Float64Array.from(T.data,v=>Number.isNaN(v)?v:Math.min(H,Math.max(L,v)));
  return track(mkT(T.shape.slice(),out,dt,T.np),[T],'ClampBackward1',g=>[Float64Array.from(g,(gv,k)=>T.data[k]>=L&&T.data[k]<=H?gv:0)])}
function varStd(T,dim,corr,keep,isStd){ // 用可求导的基本运算拼出 var / std，最后改成 torch 的 grad_fn 名字
  if(!isFloatDt(T.dtype))err('RuntimeError',`std and var only support floating point and complex dtypes`);
  const ds=dim==null?T.shape.map((_,i)=>i):normDims(T,dim),cnt=prod(ds.map(d=>T.shape[d]));
  const mu=reduce(T,dim,'mean',true),dev=binop('-',T,mu),sq=binop('*',dev,dev),sm=reduce(sq,dim,'sum',keep);
  let out=binop('/',sm,mkT([],[Math.max(0,cnt-corr)],T.dtype));if(isStd)out=unary(out,'sqrt');if(out.grad_fn)out.grad_fn.name=isStd?'StdBackward0':'VarBackward0';return out}
function reduce(T,dim,kind,keep){
  if(kind==='mean'&&!isFloatDt(T.dtype))err('RuntimeError',`mean(): could not infer output dtype. Input dtype must be either a floating point or complex dtype. Got: ${T.dtype==='int64'?'Long':'Bool'}`);
  const f=kind==='max'?Math.max:kind==='min'?Math.min:(a,b)=>a+b,init=kind==='max'?-Infinity:kind==='min'?Infinity:0;
  const dt=(kind==='sum'||kind==='prod')&&T.dtype==='bool'?'int64':T.dtype;
  const ds=dim==null?T.shape.map((_,i)=>i):normDims(T,dim);const all=dim==null;
  if(!T.data.length&&kind!=='sum'&&all)err('RuntimeError',`${kind}(): Expected reduction dim to be specified for input.numel() == 0.`);
  const outFull=T.shape.map((d,i)=>ds.includes(i)?1:d),n=prod(outFull),res=new Float64Array(n).fill(init),cnt=prod(ds.map(d=>T.shape[d]));
  const ost=strides(outFull),idx=new Array(T.shape.length).fill(0),map=new Int32Array(T.data.length);
  for(let k=0;k<T.data.length;k++){let o=0;for(let d=0;d<T.shape.length;d++)if(!ds.includes(d))o+=idx[d]*ost[d];map[k]=o;res[o]=f(res[o],T.data[k]);for(let d=T.shape.length-1;d>=0;d--){if(++idx[d]<T.shape[d])break;idx[d]=0}}
  if(kind==='mean')for(let o=0;o<n;o++)res[o]/=cnt;
  const shape=all&&!keep?[]:keep?outFull:T.shape.filter((_,i)=>!ds.includes(i));const R=mkT(shape,res,dt,T.np);
  if(!all&&T.shape.length<=2&&T.data.length<=120)R.meta={kind:'reduce',src:snapshot(T),dims:ds,op:kind,keep};
  if(kind==='sum'||kind==='mean'){const nm=kind==='sum'?(all?'SumBackward0':'SumBackward1'):(all?'MeanBackward0':'MeanBackward1');track(R,[T],nm,g=>{const r=new Float64Array(T.data.length);for(let k=0;k<r.length;k++)r[k]=g[map[k]]/(kind==='mean'?cnt:1);return[r]})}
  else track(R,[T],all?(kind==='max'?'MaxBackward1':'MinBackward1'):(kind==='max'?'MaxBackward0':'MinBackward0'),g=>{const r=new Float64Array(T.data.length),done=new Uint8Array(n);for(let k=0;k<r.length;k++)if(!done[map[k]]&&T.data[k]===res[map[k]]){r[k]=g[map[k]];done[map[k]]=1}return[r]});
  return R}
function cumsum(T,dim){const nd=T.shape.length;if(dim<0)dim+=nd;const st=strides(T.shape),d=Float64Array.from(T.data);for(let k=0;k<d.length;k++){const i=Math.floor(k/st[dim])%T.shape[dim];if(i>0)d[k]+=d[k-st[dim]]}
  const R=mkT(T.shape.slice(),d,T.dtype==='bool'?'int64':T.dtype,T.np);return track(R,[T],'CumsumBackward0',g=>{const r=Float64Array.from(g);for(let k=r.length-1;k>=0;k--){const i=Math.floor(k/st[dim])%T.shape[dim];if(i<T.shape[dim]-1)r[k]+=r[k+st[dim]]}return[r]})}
function transpose2(T){if(T.shape.length<2)return T;if(T.shape.length>2)err('RuntimeError','本页的 .T 只支持二维张量');const[m,n]=T.shape,d=new Float64Array(m*n);for(let i=0;i<m;i++)for(let j=0;j<n;j++)d[j*m+i]=T.data[i*n+j];
  return track(mkT([n,m],d,T.dtype,T.np),[T],'PermuteBackward0',g=>{const r=new Float64Array(m*n);for(let i=0;i<m;i++)for(let j=0;j<n;j++)r[i*n+j]=g[j*m+i];return[r]})}
function matmul(A,Bt,name){
  if(A.shape.length===1&&Bt.shape.length===1){if(A.shape[0]!==Bt.shape[0])err('RuntimeError',`inconsistent tensor size, expected tensor [${A.shape[0]}] and src [${Bt.shape[0]}] to have the same number of elements, but got ${A.shape[0]} and ${Bt.shape[0]} elements respectively`);
    let s=0;for(let i=0;i<A.shape[0];i++)s+=A.data[i]*Bt.data[i];return track(mkT([],[s],resultDtype(A,Bt),A.np||Bt.np),[A,Bt],'DotBackward0',g=>[Float64Array.from(Bt.data,v=>v*g[0]),Float64Array.from(A.data,v=>v*g[0])])}
  const a2=A.shape.length===1?[1,A.shape[0]]:A.shape,b2=Bt.shape.length===1?[Bt.shape[0],1]:Bt.shape;if(a2.length!==2||b2.length!==2)err('RuntimeError','本页的矩阵乘法只支持一维和二维');
  const[m,k]=a2,[k2,n]=b2;if(k!==k2)err('RuntimeError',name==='mv'?`size mismatch, got input (${m}), mat (${m}x${k}), vec (${k2})`:`mat1 and mat2 shapes cannot be multiplied (${m}x${k} and ${k2}x${n})`);
  const d=new Float64Array(m*n);for(let i=0;i<m;i++)for(let p=0;p<k;p++){const av=A.data[i*k+p];if(!av)continue;for(let j=0;j<n;j++)d[i*n+j]+=av*Bt.data[p*n+j]}
  let shape=[m,n];if(A.shape.length===1)shape=[n];else if(Bt.shape.length===1)shape=[m];
  const R=mkT(shape,d,resultDtype(A,Bt),A.np||Bt.np);if(m*k<=64&&k*n<=64)R.meta={kind:'mm',a:snapshot(A),b:snapshot(Bt)};
  return track(R,[A,Bt],Bt.shape.length===1?'MvBackward0':'MmBackward0',g=>{const ga=A.requires_grad?new Float64Array(m*k):null,gb=Bt.requires_grad?new Float64Array(k*n):null;const Ad=A.data,Bd=Bt.data;
    if(ga)for(let i=0;i<m;i++){const gi=i*n;for(let p=0;p<k;p++){let s=0;const bp=p*n;for(let j=0;j<n;j++)s+=g[gi+j]*Bd[bp+j];ga[i*k+p]=s}}
    if(gb)for(let i=0;i<m;i++){const gi=i*n,ai=i*k;for(let p=0;p<k;p++){const a=Ad[ai+p];if(!a)continue;const bp=p*n;for(let j=0;j<n;j++)gb[bp+j]+=a*g[gi+j]}}return[ga,gb]})}
const UN={exp:[Math.exp,'ExpBackward0',(a,o)=>o],log:[Math.log,'LogBackward0',a=>1/a],log2:[Math.log2,'Log2Backward0',a=>1/(a*Math.LN2)],log10:[Math.log10,'Log10Backward0',a=>1/(a*Math.LN10)],sqrt:[Math.sqrt,'SqrtBackward0',(a,o)=>.5/o],sin:[Math.sin,'SinBackward0',a=>Math.cos(a)],cos:[Math.cos,'CosBackward0',a=>-Math.sin(a)],
  tanh:[Math.tanh,'TanhBackward0',(a,o)=>1-o*o],sigmoid:[a=>1/(1+Math.exp(-a)),'SigmoidBackward0',(a,o)=>o*(1-o)],relu:[a=>Math.max(0,a),'ReluBackward0',a=>a>0?1:0],abs:[Math.abs,'AbsBackward0',a=>Math.sign(a)],neg:[a=>-a,'NegBackward0',()=>-1],square:[a=>a*a,'PowBackward0',a=>2*a]};
function unary(x,name){const X=asTensor(x),[f,bn,df]=UN[name];const keep=name==='abs'||name==='neg'||name==='relu'||name==='square';const dt=keep?X.dtype:(isFloatDt(X.dtype)?X.dtype:'float32');
  const out=Float64Array.from(X.data,f);const R=mkT(X.shape.slice(),out,dt,X.np);return track(R,[X],bn,g=>[Float64Array.from(g,(gv,k)=>gv*df(X.data[k],R.data[k]))])}
function normT(T){if(!isFloatDt(T.dtype))err('RuntimeError',`linalg.vector_norm: Expected a floating point or complex tensor as input. Got ${T.dtype==='int64'?'Long':'Bool'}`);let s=0;for(const v of T.data)s+=v*v;const o=Math.sqrt(s);
  return track(mkT([],[o],T.dtype,T.np),[T],'LinalgVectorNormBackward0',g=>[Float64Array.from(T.data,v=>o?g[0]*v/o:0)])}

function invT(T){if(T.shape.length!==2||T.shape[0]!==T.shape[1])err('RuntimeError','linalg.inv: A must be batches of square matrices, but they are '+T.shape.join(' by ')+' matrices');const n=T.shape[0],A=[];for(let i=0;i<n;i++){const r=new Float64Array(2*n);for(let j=0;j<n;j++)r[j]=T.data[i*n+j];r[n+i]=1;A.push(r)}
  for(let c=0;c<n;c++){let p=c;for(let r=c+1;r<n;r++)if(Math.abs(A[r][c])>Math.abs(A[p][c]))p=r;if(Math.abs(A[p][c])<1e-12)err('RuntimeError','linalg.inv: The diagonal element '+(c+1)+' is zero, the inversion could not be completed because the input matrix is singular.');[A[c],A[p]]=[A[p],A[c]];const d=A[c][c];for(let j=0;j<2*n;j++)A[c][j]/=d;for(let r=0;r<n;r++){if(r===c)continue;const f=A[r][c];if(f)for(let j=0;j<2*n;j++)A[r][j]-=f*A[c][j]}}
  const out=[];for(let i=0;i<n;i++)for(let j=0;j<n;j++)out.push(A[i][n+j]);return mkT([n,n],out,T.dtype)}

function argmaxT(T,dim,keep,sign){if(dim==null){let b=0;T.data.forEach((v,i)=>{if(sign*v>sign*T.data[b])b=i});return mkT([],[b],'int64')}
  const G=groups(T,dim),nd=T.shape.length,dd=dim<0?dim+nd:dim;const out=G.map(g=>{let b=0;g.forEach((o,i)=>{if(sign*T.data[o]>sign*T.data[g[b]])b=i});return b});const shape=T.shape.filter((_,i)=>i!==dd);const R=mkT(shape,out,'int64');return keep?reshapeT(R,T.shape.map((d2,i)=>i===dd?1:d2)):R}

function maxmin(x,y,sign){const A=asTensor(x),Bt=asTensor(y),out=bshape(A.shape,Bt.shape),n=prod(out),sa=bidx(A.shape,out),sb=bidx(Bt.shape,out),res=new Float64Array(n),pick=new Uint8Array(n),idx=new Array(out.length).fill(0);
  for(let k=0;k<n;k++){let ia=0,ib=0;for(let d=0;d<out.length;d++){ia+=idx[d]*sa[d];ib+=idx[d]*sb[d]}const a=A.data[ia],b=Bt.data[ib];const ta=sign>0?a>b:a<b;res[k]=ta?a:b;pick[k]=a===b?2:ta?0:1;for(let d=out.length-1;d>=0;d--){if(++idx[d]<out[d])break;idx[d]=0}}
  const R=mkT(out,res,resultDtype(A,Bt));return track(R,[A,Bt],sign>0?'MaximumBackward0':'MinimumBackward0',g=>{const ga=new Float64Array(n),gb=new Float64Array(n);for(let k=0;k<n;k++){if(pick[k]===0)ga[k]=g[k];else if(pick[k]===1)gb[k]=g[k];else{ga[k]=g[k]/2;gb[k]=g[k]/2}}return[unb(ga,out,A.shape),unb(gb,out,Bt.shape)]})}

// PyTorch 的 arange 用乘加融合（FMA）算 start + i*step，这里精确模拟
function splitD(a){const t=134217729*a,hi=t-(t-a);return[hi,a-hi]}
function fmaD(a,b,c){const p=a*b;if(!Number.isFinite(p))return p+c;const[ah,al]=splitD(a),[bh,bl]=splitD(b);const e=((ah*bh-p)+ah*bl+al*bh)+al*bl;const s=p+c,bb=s-p,es=(p-(s-bb))+(c-bb);return s+(es+e)}

function gammaFn(x){if(Number.isInteger(x)&&x>0&&x<171){let r=1;for(let i=2;i<x;i++)r*=i;return r}if(x<0.5)return Math.PI/(Math.sin(Math.PI*x)*gammaFn(1-x));x-=1;const g=7,c=[0.99999999999980993,676.5203681218851,-1259.1392167224028,771.32342877765313,-176.61502916214059,12.507343278686905,-0.13857109526572012,9.9843695780195716e-6,1.5056327351493116e-7];let a=c[0];const t=x+g+0.5;for(let i=1;i<9;i++)a+=c[i]/(x+i);return Math.sqrt(2*Math.PI)*t**(x+0.5)*Math.exp(-t)*a}
