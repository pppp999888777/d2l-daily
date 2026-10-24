
// ---------- 索引 ----------
function selection(T,items){const nd=T.shape.length;let k=0;const sel=[];const nonEll=items.filter(it=>!it.ellipsis&&!it.none).length;
  for(const it of items){if(it.none){sel.push({newaxis:true});continue}if(it.ellipsis){const rep=nd-nonEll;for(let r=0;r<rep;r++){sel.push({idx:[...Array(T.shape[k]).keys()],keep:true});k++}continue}
    if(k>=nd)err('IndexError',`too many indices for tensor of dimension ${nd}`);const n=T.shape[k];
    if(it.slice){let[a,b,c]=it.slice;c=c==null?1:c;if(c<=0)err('ValueError','step must be greater than zero');const norm=(x,def)=>{if(x==null)return def;if(x<0)x+=n;return Math.min(Math.max(x,0),n)};a=norm(a,0);b=norm(b,n);const idx=[];for(let i=a;i<b;i+=c)idx.push(i);sel.push({idx,keep:true,slice:true})}
    else if(it.tensor){const idx=[...it.tensor.data].map(i=>{if(i<-n||i>=n)err('IndexError',`index ${i} is out of bounds for dimension ${k} with size ${n}`);return i<0?i+n:i});sel.push({idx,keep:true,adv:true})}
    else{let i=it.v;if(i<-n||i>=n)err('IndexError',`index ${i} is out of bounds for dimension ${k} with size ${n}`);if(i<0)i+=n;sel.push({idx:[i],keep:false})}k++}
  for(;k<nd;k++)sel.push({idx:[...Array(T.shape[k]).keys()],keep:true});return sel}
function forEachSel(T,sel,fn){const s=sel.filter(x=>!x.newaxis),st=strides(T.shape),nd=s.length;if(nd===0){fn(0,0);return}if(s.some(x=>!x.idx.length))return;const pos=new Array(nd).fill(0);let c=0;
  while(true){let off=0;for(let d=0;d<nd;d++)off+=s[d].idx[pos[d]]*st[d];fn(off,c++);let d=nd-1;for(;d>=0;d--){if(++pos[d]<s[d].idx.length)break;pos[d]=0}if(d<0)break}}
function selShape(sel){return sel.flatMap(s=>s.newaxis?[1]:s.keep?[s.idx.length]:[])}
function pairedGather(T,items){const advs=items.filter(it=>it.tensor);const L=advs[0].tensor.data.length;if(advs.some(a=>a.tensor.data.length!==L))err('IndexError',`shape mismatch: indexing tensors could not be broadcast together with shapes ${advs.map(a=>'['+a.tensor.data.length+']').join(', ')}`);
  if(items.length!==advs.length||items.length>T.shape.length)err('IndexError','本页只支持“每个轴都给一个下标列表”的配对索引');const st=strides(T.shape),offs=[];
  for(let k=0;k<L;k++){let off=0;items.forEach((it,d)=>{let i=it.tensor.data[k];const n=T.shape[d];if(i<-n||i>=n)err('IndexError',`index ${i} is out of bounds for dimension ${d} with size ${n}`);if(i<0)i+=n;off+=i*st[d]});offs.push(off)}
  const rest=T.shape.slice(items.length),rn=prod(rest),vals=[];for(const o of offs)for(let r=0;r<rn;r++)vals.push(T.data[o+r]);const R=mkT([L,...rest],vals,T.dtype,T.np);
  if(T.shape.length<=3&&T.data.length<=300){const mask=new Uint8Array(T.data.length);offs.forEach(o=>{for(let r=0;r<rn;r++)mask[o+r]=1});R.meta={kind:'index',src:snapshot(T),mask}}
  return track(R,[T],'IndexBackward0',g=>{const r=new Float64Array(T.data.length);offs.forEach((o,k)=>{for(let q=0;q<rn;q++)r[o+q]+=g[k*rn+q]});return[r]})}
function getItem(T,items){if(items.filter(it=>it.tensor).length>=2)return pairedGather(T,items);if(items.length===1&&items[0].bool){const M=items[0].bool;const vals=[];T.data.forEach((v,i)=>{if(M.data[i])vals.push(v)});return mkT([vals.length],vals,T.dtype,T.np)}
  const sel=selection(T,items),shape=selShape(sel),vals=[],offs=[],mask=new Uint8Array(T.data.length);
  forEachSel(T,sel,off=>{vals.push(T.data[off]);offs.push(off);mask[off]=1});const R=mkT(shape,vals,T.dtype,T.np);
  if(T.shape.length>=1&&T.shape.length<=3&&T.data.length<=300)R.meta={kind:'index',src:snapshot(T),mask};
  // grad_fn 的名字取最后一个真正起作用的索引操作（和 PyTorch 逐个应用下标一致）：x[:, 0] 是 Select，x[0, 1:3] 是 Slice
  let last=null;sel.forEach((s,d)=>{if(s.newaxis)return;if(!s.keep)last='SelectBackward0';else if(s.slice&&!(s.idx.length===T.shape[d]&&s.idx.every((v,i)=>v===i)))last='SliceBackward0'});
  const nm=sel.some(s=>s.adv)?'IndexBackward0':last||(sel.some(s=>s.slice)?'SliceBackward0':'AliasBackward0');
  return track(R,[T],nm,g=>{const r=new Float64Array(T.data.length);offs.forEach((o,c)=>r[o]+=g[c]);return[r]})}
function setItem(T,items,v){checkInplace(T);if(items.length===1&&items[0].bool){const M=items[0].bool,x=numOf(v);T.data.forEach((_,i)=>{if(M.data[i])T.data[i]=castVal(x,T.dtype)});return}
  const sel=selection(T,items),shape=selShape(sel),mask=new Uint8Array(T.data.length);let src=null,x=0,ibs=null;
  if(isT(v)){const V=v;bshape(shape,V.shape);const n=prod(shape),sb=bidx(V.shape,shape),idx=new Array(shape.length).fill(0);src=new Float64Array(n);ibs=new Int32Array(n);
    for(let k=0;k<n;k++){let ib=0;for(let d=0;d<shape.length;d++)ib+=idx[d]*sb[d];src[k]=V.data[ib];ibs[k]=ib;for(let d=shape.length-1;d>=0;d--){if(++idx[d]<shape[d])break;idx[d]=0}}}else x=numOf(v);
  const pos=[];forEachSel(T,sel,(off,c)=>{T.data[off]=castVal(src?src[c]:x,T.dtype);mask[off]=1;pos[c]=off});T.meta={kind:'assign',mask};
  // 自动求导：往张量里赋值一个需要梯度的值（或者给已经在计算图里的张量赋值），记成 CopySlices
  const V=isT(v)?v:null;if(gradEnabled&&(T.requires_grad||(V&&V.requires_grad))){const prev=mkT(T.shape.slice(),[],T.dtype);prev.requires_grad=T.requires_grad;prev.grad_fn=T.grad_fn;T.requires_grad=false;T.grad_fn=null;
    track(T,V?[prev,V]:[prev],'CopySlices',g=>{const gp=Float64Array.from(g);for(const off of pos)gp[off]=0;if(!V)return[gp];const gv=new Float64Array(V.data.length);pos.forEach((off,c)=>{gv[ibs[c]]+=g[off]});return[gp,gv]})}}

// ---------- 模拟文件系统 ----------
const FS=new Map();const DIRS=new Set(['.','..']);
// 网页里内置的数据文件（比如时光机器文本）：重置文件系统后还在
const STATIC_FILES=new Map();
function addStatic(){for(const[p,t]of STATIC_FILES){FS.set(p,t);let d=p;while(d.includes('/')){d=d.slice(0,d.lastIndexOf('/'));DIRS.add(d)}}}
function setStaticFile(path,text){STATIC_FILES.set(normPath(path),text);addStatic()}
function normPath(p){const parts=[];for(const s of p.split('/')){if(s===''||s==='.')continue;if(s==='..'){if(parts.length&&parts[parts.length-1]!=='..')parts.pop();else parts.push('..')}else parts.push(s)}return parts.join('/')||'.'}
function openFile(path,mode){const p=normPath(path);mode=mode||'r';
  if(mode.startsWith('r')){if(!FS.has(p))err('FileNotFoundError',`[Errno 2] No such file or directory: '${path}'`);return{t:'file',path,p,mode,pos:0}}
  const dir=p.includes('/')?p.slice(0,p.lastIndexOf('/')):'.';if(!DIRS.has(dir)&&dir!=='.')err('FileNotFoundError',`[Errno 2] No such file or directory: '${path}'`);
  if(mode.startsWith('w'))FS.set(p,'');else if(!FS.has(p))FS.set(p,'');return{t:'file',path,p,mode,pos:0}}

// ---------- 小型 pandas ----------
const NA_TOK=new Set(['','NA','NaN','nan','N/A','null','NULL']);
const isMissing=(v,dt)=>dt==='str'?v==null:(v==null||Number.isNaN(v));
function inferCol(raw){const vals=raw.map(s=>NA_TOK.has(s)?null:s);const nn=vals.filter(v=>v!=null);
  if(nn.length&&nn.every(s=>s==='True'||s==='False'))return{dt:'bool',v:vals.map(s=>s==null?NaN:s==='True')};
  if(nn.every(s=>/^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$/.test(s.trim()))){const nums=vals.map(s=>s==null?NaN:+s);const allInt=nn.every(s=>/^[-+]?\d+$/.test(s.trim()));return{dt:allInt&&nn.length===vals.length?'int64':'float64',v:nums}}
  return{dt:'str',v:vals}}
function makeDF(cols,data,dtype,index){return{t:'df',cols,data,dtype,index:index||data[cols[0]]?.map((_,i)=>i)||[]}}
function readCSV(path){const f=openFile(path,'r');const text=FS.get(f.p).replace(/\r/g,'');const lines=text.split('\n').filter(l=>l.length);if(!lines.length)err('EmptyDataError','No columns to parse from file');
  const head=lines[0].split(',');const rows=lines.slice(1).map(l=>l.split(','));const data={},dtype={};
  head.forEach((h,j)=>{const c=inferCol(rows.map(r=>r[j]??''));data[h]=c.v;dtype[h]=c.dt});return makeDF(head,data,dtype,rows.map((_,i)=>i))}
function dfFromDict(d){const cols=[],data={},dtype={};for(const[k,v]of d.m.values()){const name=str(k);cols.push(name);const items=isT(v)?[...v.data].map(x=>({t:isFloatDt(v.dtype)?'float':'int',v:x})):iterate(v);
  const c=pyListCol(items);data[name]=c.v;dtype[name]=c.dt}return makeDF(cols,data,dtype)}
function pyListCol(items){if(items.every(x=>x.t==='bool'))return{dt:'bool',v:items.map(x=>x.v)};if(items.every(x=>x.t==='int'))return{dt:'int64',v:items.map(x=>x.v)};
  if(items.every(x=>x.t==='int'||x.t==='float'||x.t==='none'||x.t==='bool'))return{dt:'float64',v:items.map(x=>x.t==='none'?NaN:+pyNum(x))};return{dt:'str',v:items.map(x=>x.t==='none'?null:str(x))}}
function floatDecimals(vals){let d=1;for(const v of vals){if(!Number.isFinite(v))continue;const s=String(+v.toFixed(6)),p=s.indexOf('.');if(p>=0)d=Math.max(d,s.length-p-1)}return Math.min(d,6)}
function fmtCol(vals,dt){if(dt==='str')return vals.map(v=>' '+(v==null?'NaN':v));if(dt==='bool')return vals.map(v=>Number.isNaN(v)?' NaN':' '+(v?'True':'False'));
  if(dt==='int64')return vals.map(v=>(v<0?'':' ')+String(v));const d=floatDecimals(vals);return vals.map(v=>Number.isNaN(v)?' NaN':(v<0?'':' ')+v.toFixed(d))}
function dfRepr(df){if(!df.index.length||!df.cols.length)return`Empty DataFrame\nColumns: [${df.cols.join(', ')}]\nIndex: [${df.index.join(', ')}]`;
  const idx=df.index.map(String),wi=Math.max(...idx.map(s=>s.length));const cols=df.cols.map(c=>{const f=fmtCol(df.data[c],df.dtype[c]);const L=Math.max(...f.map(s=>s.length));const W=Math.max(String(c).length+(df.dtype[c]==='str'?0:1),L)+1;return{h:String(c).padStart(W),f:f.map(s=>s.padStart(W))}});
  const lines=[' '.repeat(wi)+cols.map(c=>c.h).join('')];idx.forEach((s,i)=>lines.push(s.padEnd(wi)+cols.map(c=>c.f[i]).join('')));return lines.join('\n')}
function seriesRepr(s){const dtn=s.dtype;const f=fmtCol(s.values,s.dtype==='object'?'str':s.dtype);const idx=s.index.map(String),wi=Math.max(0,...idx.map(x=>x.length)),L=Math.max(0,...f.map(x=>x.length));
  const lines=idx.map((x,i)=>x.padEnd(wi)+f[i].padStart(L+3));lines.push((s.name!=null?`Name: ${s.name}, `:'')+`dtype: ${dtn}`);return lines.join('\n')}
const mkSeries=(values,dtype,index,name)=>({t:'series',values,dtype,index,name:name??null});
function colSeries(df,c){return mkSeries(df.data[c].slice(),df.dtype[c],df.index.slice(),c)}
function dfSelectCols(df,cs){const data={},dtype={};for(const c of cs){data[c]=df.data[c].slice();dtype[c]=df.dtype[c]}return makeDF(cs,data,dtype,df.index.slice())}
function dfSelectRows(df,rows){const data={};for(const c of df.cols)data[c]=rows.map(i=>df.data[c][i]);return makeDF(df.cols.slice(),data,{...df.dtype},rows.map(i=>df.index[i]))}
function sliceIdx(it,n){if(it.slice){let[a,b,c]=it.slice;c=c||1;const norm=(x,d)=>x==null?d:x<0?Math.max(0,x+n):Math.min(x,n);a=norm(a,0);b=norm(b,n);const r=[];for(let i=a;i<b;i+=c)r.push(i);return{list:r,scalar:false}}let i=it.v;if(i<0)i+=n;if(i<0||i>=n)err('IndexError','single positional indexer is out-of-bounds');return{list:[i],scalar:true}}
function ilocGet(df,items){const r=sliceIdx(items[0],df.index.length),c=items.length>1?sliceIdx(items[1],df.cols.length):{list:df.cols.map((_,i)=>i),scalar:false};
  const cols=c.list.map(j=>df.cols[j]);
  if(r.scalar&&c.scalar)return pyVal(df.data[cols[0]][r.list[0]],df.dtype[cols[0]]);
  if(c.scalar){const s=colSeries(df,cols[0]);s.values=r.list.map(i=>s.values[i]);s.index=r.list.map(i=>df.index[i]);return r.scalar?pyVal(s.values[0],s.dtype):s}
  const sub=dfSelectRows(dfSelectCols(df,cols),r.list);if(r.scalar){const row=r.list[0];return mkSeries(cols.map(cn=>df.data[cn][row]),cols.every(cn=>df.dtype[cn]!=='str')?'float64':'object',cols,df.index[row])}return sub}
function pyVal(v,dt){if(dt==='str'||dt==='object')return v==null?F(NaN):S(String(v));if(dt==='bool')return B(v);if(dt==='int64')return I(v);return F(v)}
const numericDt=dt=>dt==='int64'||dt==='float64'||dt==='bool';
function dfMean(df,numericOnly){const cs=[];for(const c of df.cols){if(!numericDt(df.dtype[c])){if(numericOnly)continue;err('TypeError',`Cannot perform reduction 'mean' with string dtype`)}cs.push(c)}
  return mkSeries(cs.map(c=>{const v=df.data[c].filter(x=>!Number.isNaN(+x)).map(Number);return v.length?v.reduce((a,b)=>a+b,0)/v.length:NaN}),'float64',cs)}
function dfFillna(df,val){const data={},dtype={...df.dtype};for(const c of df.cols){let fill=null;if(val.t==='series'){const k=val.index.indexOf(c);if(k>=0)fill=val.values[k]}else fill=val.t==='str'?val.v:pyNum(val);
  data[c]=df.data[c].map(x=>isMissing(x,df.dtype[c])&&fill!=null&&!(typeof fill==='number'&&Number.isNaN(fill))?fill:x);if(fill!=null&&df.dtype[c]==='str'&&typeof fill==='number')dtype[c]='object'}return makeDF(df.cols.slice(),data,dtype,df.index.slice())}
function getDummies(df,kw){const dummyNa=kw.dummy_na&&kw.dummy_na.v;const keep=[],dums=[];const data={},dtype={};
  for(const c of df.cols){if(df.dtype[c]==='str'||df.dtype[c]==='object'){const cats=[...new Set(df.data[c].filter(v=>v!=null))].sort();for(const k of cats){const n=`${c}_${k}`;dums.push(n);data[n]=df.data[c].map(v=>v===k);dtype[n]='bool'}
      if(dummyNa){const n=`${c}_nan`;dums.push(n);data[n]=df.data[c].map(v=>v==null);dtype[n]='bool'}}else{keep.push(c);data[c]=df.data[c].slice();dtype[c]=df.dtype[c]}}
  return makeDF([...keep,...dums],data,dtype,df.index.slice())}
function dfToNumpy(df,kw){const dt=kw.dtype?dtypeKw(kw,'float64'):null;const cs=df.cols;const anyStr=cs.some(c=>!numericDt(df.dtype[c]));
  if(anyStr&&!dt)return{t:'objarr'};const vals=[];for(let i=0;i<df.index.length;i++)for(const c of cs){const v=df.data[c][i];if(typeof v==='string')err('ValueError',`could not convert string to float: '${v}'`);vals.push(v==null?NaN:+v)}
  const allBool=cs.every(c=>df.dtype[c]==='bool'),allInt=cs.every(c=>df.dtype[c]==='int64');return mkT([df.index.length,cs.length],vals,dt||(allBool?'bool':allInt?'int64':'float64'),true)}
function seriesToNumpy(s,kw){const dt=kw&&kw.dtype?dtypeKw(kw,'float64'):null;if(s.dtype==='str'&&!dt)return{t:'objarr'};return mkT([s.values.length],s.values.map(v=>{if(typeof v==='string')err('ValueError',`could not convert string to float: '${v}'`);return v==null?NaN:+v}),dt||(s.dtype==='bool'?'bool':s.dtype==='int64'?'int64':'float64'),true)}
function seriesBinop(op,x,y){const s=x.t==='series'?x:y,o=x.t==='series'?y:x;const ov=i=>o.t==='series'?+o.values[i]:pyNum(o);
  const vals=s.values.map((v,i)=>{const r=x.t==='series'?OPS[op](+v,ov(i)):OPS[op](ov(i),+v);return CMP.has(op)?!!r:r});
  const dt=CMP.has(op)?'bool':(s.dtype==='int64'&&op!=='/'&&vals.every(Number.isInteger))?'int64':'float64';return mkSeries(vals,dt,s.index.slice(),s.name)}
function dfAttr(df,name){switch(name){
  case'shape':return tup([I(df.index.length),I(df.cols.length)]);case'columns':return{t:'index',items:df.cols.slice()};case'index':return list(df.index.map(I));
  case'iloc':return{t:'iloc',df};case'values':return dfToNumpy(df,{});
  case'dtypes':return mkSeries(df.cols.map(c=>df.dtype[c]==='str'?'str':df.dtype[c]),'object',df.cols.slice())}
  const M={mean:(a,kw)=>dfMean(df,kw.numeric_only&&kw.numeric_only.v),fillna:(a,kw)=>dfFillna(df,a[0]||kw.value),
    isna:()=>{const data={},dtype={};for(const c of df.cols){data[c]=df.data[c].map(v=>isMissing(v,df.dtype[c]));dtype[c]='bool'}return makeDF(df.cols.slice(),data,dtype,df.index.slice())},
    sum:()=>mkSeries(df.cols.map(c=>df.data[c].reduce((a,b)=>a+(Number.isNaN(+b)?0:+b),0)),df.cols.every(c=>df.dtype[c]==='bool'||df.dtype[c]==='int64')?'int64':'float64',df.cols.slice()),
    head:a=>dfSelectRows(df,df.index.map((_,i)=>i).slice(0,a[0]?intOf(a[0]):5)),
    drop:(a,kw)=>{let cs=kw.columns||(kw.axis&&pyNum(kw.axis)===1?a[0]:null);if(!cs)err('KeyError','本页的 drop 只支持删除列：drop(columns=[...]) 或 drop(列名, axis=1)');const names=(cs.items?cs.items:[cs]).map(str);for(const n of names)if(!df.cols.includes(n))err('KeyError',`"['${n}'] not found in axis"`);return dfSelectCols(df,df.cols.filter(c=>!names.includes(c)))},
    dropna:(a,kw)=>{const how=kw.how?kw.how.v:'any';if(kw.axis&&pyNum(kw.axis)===1){return dfSelectCols(df,df.cols.filter(c=>{const m=df.data[c].map(v=>isMissing(v,df.dtype[c]));return how==='all'?!m.every(Boolean):!m.some(Boolean)}))}
      const rows=df.index.map((_,i)=>i).filter(i=>{const m=df.cols.map(c=>isMissing(df.data[c][i],df.dtype[c]));return how==='all'?!m.every(Boolean):!m.some(Boolean)});return dfSelectRows(df,rows)},
    to_numpy:(a,kw)=>dfToNumpy(df,kw),copy:()=>dfSelectCols(df,df.cols)};
  if(M[name])return fn(name,M[name]);if(df.cols.includes(name))return colSeries(df,name);err('AttributeError',`'DataFrame' object has no attribute '${name}'`)}
function seriesAttr(s,name){switch(name){case'name':return s.name==null?NONE:S(String(s.name));case'values':return seriesToNumpy(s);case'shape':return tup([I(s.values.length)]);case'index':return{t:'index',items:s.index.map(String)}}
  const num=()=>s.values.filter(v=>!Number.isNaN(+v)&&v!=null).map(Number);
  const M={to_numpy:(a,kw)=>seriesToNumpy(s,kw),mean:()=>F(num().reduce((a,b)=>a+b,0)/num().length),sum:()=>{const v=num().reduce((a,b)=>a+b,0);return s.dtype==='float64'?F(v):I(v)},max:()=>pyVal(Math.max(...num()),s.dtype),min:()=>pyVal(Math.min(...num()),s.dtype),
    idxmax:()=>{const v=s.values.map(Number);let b=0;v.forEach((x,i)=>{if(x>v[b])b=i});const k=s.index[b];return typeof k==='number'?I(k):S(k)},
    isna:()=>mkSeries(s.values.map(v=>isMissing(v,s.dtype)),'bool',s.index.slice(),s.name),fillna:a=>{const f=a[0];return mkSeries(s.values.map(v=>isMissing(v,s.dtype)?(f.t==='str'?f.v:pyNum(f)):v),s.dtype,s.index.slice(),s.name)},
    tolist:()=>list(s.values.map(v=>pyVal(v,s.dtype))),value_counts:()=>{const m=new Map();for(const v of s.values)if(v!=null)m.set(v,(m.get(v)||0)+1);const e=[...m.entries()].sort((a,b)=>b[1]-a[1]);return mkSeries(e.map(x=>x[1]),'int64',e.map(x=>x[0]),'count')},
    unique:()=>list([...new Set(s.values)].map(v=>pyVal(v,s.dtype)))};
  if(M[name])return fn(name,M[name]);err('AttributeError',`'Series' object has no attribute '${name}'`)}
