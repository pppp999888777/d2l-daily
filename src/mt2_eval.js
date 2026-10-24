
// ---------- 语法分析 ----------
function parse(src){const toks=tokenize(src);let p=0;const peek=(o=0)=>toks[p+o],next=()=>toks[p++];
  const isOp=v=>peek().type==='op'&&peek().v===v,isName=v=>peek().type==='name'&&peek().v===v;
  const expect=v=>{if(!isOp(v))err('SyntaxError',`第 ${peek().line} 行：这里应该是「${v}」${peek().type==='nl'?'，但这一行已经结束了':''}`);return next()};
  const expectName=v=>{if(!isName(v))err('SyntaxError',`第 ${peek().line} 行：这里应该是「${v}」`);return next()};
  const KW=new Set(['if','else','elif','for','while','in','not','and','or','def','return','lambda','with','as','import','from','pass','break','continue','is','None','True','False','yield','class','global','nonlocal','assert','del','try','except','finally','raise']);
  function atom(){const t=next();
    if(t.type==='num')return{k:'num',v:t.v,fl:t.fl};
    if(t.type==='str'||t.type==='fstr'){const parts=[],add=x=>{if(x.type==='str')parts.push(x.v);else parts.push(...fparts(x.raw,x.line))};add(t);let f=t.type==='fstr';while(peek().type==='str'||peek().type==='fstr'){const u=next();if(u.type==='fstr')f=true;add(u)}return f?{k:'fstr',parts}:{k:'str',v:parts.join('')}}
    if(t.type==='name'){if(t.v==='None')return{k:'const',v:NONE};if(t.v==='True')return{k:'const',v:B(true)};if(t.v==='False')return{k:'const',v:B(false)};
      if(KW.has(t.v))err('SyntaxError',`第 ${t.line} 行：「${t.v}」是关键字，不能放在这里`);return{k:'name',v:t.v,line:t.line}}
    if(t.type==='op'){
      if(t.v==='('){if(isOp(')')){next();return{k:'tuple',items:[]}}const first=starOrExpr();if(isName('for')){const c=compFor(first);expect(')');return{k:'gen',...c}}
        if(isOp(',')){const items=[first];while(isOp(',')){next();if(isOp(')'))break;items.push(starOrExpr())}expect(')');return{k:'tuple',items}}expect(')');return first}
      if(t.v==='['){if(isOp(']')){next();return{k:'list',items:[]}}const first=starOrExpr();if(isName('for')){const c=compFor(first);expect(']');return{k:'listcomp',...c}}const items=[first];while(isOp(',')){next();if(isOp(']'))break;items.push(starOrExpr())}expect(']');return{k:'list',items}}
      if(t.v==='{'){if(isOp('}')){next();return{k:'dict',pairs:[]}}const k=expr();if(isOp(':')){next();const v=expr();if(isName('for')){const c=compFor(null);expect('}');return{k:'dictcomp',key:k,val:v,...c}}const pairs=[[k,v]];while(isOp(',')){next();if(isOp('}'))break;const k2=expr();expect(':');pairs.push([k2,expr()])}expect('}');return{k:'dict',pairs}}
        const items=[k];while(isOp(',')){next();if(isOp('}'))break;items.push(expr())}expect('}');return{k:'setlit',items}}
      if(t.v==='...')return{k:'const',v:{t:'ellipsis'}}}
    err('SyntaxError',`第 ${t.line} 行：这里不应该出现「${t.v??(t.type==='nl'?'换行':t.type==='indent'?'缩进':t.type)}」`)}
  function compFor(elt){const clauses=[];while(isName('for')||isName('if')){if(isName('for')){next();const target=targetList();expectName('in');clauses.push({k:'for',target,iter:orTest()})}else{next();clauses.push({k:'if',cond:orTest()})}}return{elt,clauses}}
  function fparts(raw,line){const parts=[];let i=0,lit='';while(i<raw.length){const c=raw[i];
    if(c==='{'&&raw[i+1]==='{'){lit+='{';i+=2;continue}if(c==='}'&&raw[i+1]==='}'){lit+='}';i+=2;continue}
    if(c==='{'){if(lit){parts.push(lit);lit=''}let d=0,j=i+1,q=null,exprEnd=-1,specStart=-1,conv=null;for(;j<raw.length;j++){const ch=raw[j];if(q){if(ch===q)q=null;continue}if(ch==="'"||ch==='"'){q=ch;continue}
        if('([{'.includes(ch))d++;else if(')]'.includes(ch))d--;else if(ch==='}'){if(d===0)break;d--}else if(d===0&&ch==='!'&&raw[j+1]!=='='&&exprEnd<0){exprEnd=j;conv=raw[j+1];j++}else if(d===0&&ch===':'&&specStart<0){if(exprEnd<0)exprEnd=j;specStart=j+1}}
      if(j>=raw.length)err('SyntaxError',`第 ${line} 行：f-string 里的 {{ 没有对应的 }}`);let ex=raw.slice(i+1,exprEnd<0?j:exprEnd),eqDbg=false;if(/=\s*$/.test(ex)&&!/[=!<>]=\s*$/.test(ex)){eqDbg=ex;ex=ex.replace(/=\s*$/,'')}
      parts.push({ast:parseExprOnly(ex,line),spec:specStart<0?'':raw.slice(specStart,j),conv,dbg:eqDbg});i=j+1;continue}
    lit+=c;i++}if(lit)parts.push(lit);return parts}
  function postfix(){let e=atom();for(;;){
    if(isOp('(')){next();const args=[],kw=[];while(!isOp(')')){if(isOp('*')){next();args.push({star:expr()})}else if(isOp('**')){next();kw.push({dstar:expr()})}
        else if(peek().type==='name'&&peek(1).type==='op'&&peek(1).v==='='){const k=next().v;next();kw.push({k,v:expr()})}else{const a=expr();if(isName('for')){const c=compFor(a);args.push({k:'gen',...c})}else args.push(a)}if(isOp(','))next();else break}
      expect(')');e={k:'call',fn:e,args,kw,line:peek().line}}
    else if(isOp('[')){next();const items=[];while(!isOp(']')){items.push(subItem());if(isOp(','))next();else break}expect(']');e={k:'sub',obj:e,items}}
    else if(isOp('.')){next();const t=next();if(t.type!=='name')err('SyntaxError',`第 ${t.line} 行：点号后面要跟名字`);e={k:'attr',obj:e,name:t.v}}
    else return e}}
  function subItem(){if(isOp('...')){next();return{ellipsis:true}}let a=null,b=null,c=null;if(!isOp(':'))a=expr();if(!isOp(':'))return{e:a};next();if(!isOp(':')&&!isOp(',')&&!isOp(']'))b=expr();if(isOp(':')){next();if(!isOp(',')&&!isOp(']'))c=expr()}return{slice:[a,b,c]}}
  function power(){const b=postfix();if(isOp('**')){next();return{k:'bin',op:'**',a:b,b:unaryE()}}return b}
  function unaryE(){if(isOp('-')){next();return{k:'neg',a:unaryE()}}if(isOp('+')){next();return unaryE()}if(isOp('~')){next();return{k:'inv',a:unaryE()}}return power()}
  function mul(){let e=unaryE();while(['*','/','//','%','@'].some(isOp)){const op=next().v;e={k:'bin',op,a:e,b:unaryE()}}return e}
  function add(){let e=mul();while(isOp('+')||isOp('-')){const op=next().v;e={k:'bin',op,a:e,b:mul()}}return e}
  function band(){let e=add();while(isOp('&')){next();e={k:'bin',op:'&',a:e,b:add()}}return e}
  function bor(){let e=band();while(isOp('|')){next();e={k:'bin',op:'|',a:e,b:band()}}return e}
  function cmp(){const first=bor();const ops=[],rest=[];for(;;){let op=null;if(['==','!=','<','>','<=','>='].some(isOp))op=next().v;else if(isName('in')){next();op='in'}else if(isName('not')&&peek(1).type==='name'&&peek(1).v==='in'){next();next();op='not in'}else if(isName('is')){next();if(isName('not')){next();op='is not'}else op='is'}
      if(!op)break;ops.push(op);rest.push(bor())}return ops.length?{k:'cmp',first,ops,rest}:first}
  function notE(){if(isName('not')){next();return{k:'not',a:notE()}}return cmp()}
  function andE(){let e=notE();while(isName('and')){next();e={k:'and',a:e,b:notE()}}return e}
  function orTest(){let e=andE();while(isName('or')){next();e={k:'or',a:e,b:andE()}}return e}
  function expr(){if(isName('lambda')){next();const params=paramList(':');expect(':');return{k:'lambda',params,body:expr()}}const e=orTest();if(isName('if')&&!inCompIf){next();const c=orTest();expectName('else');return{k:'ifexp',c,a:e,b:expr()}}return e}
  let inCompIf=false;
  function starOrExpr(){if(isOp('*')){next();return{k:'starred',e:bor()}}return expr()}
  function tupleExpr(){const first=starOrExpr();if(!isOp(','))return first;const items=[first];while(isOp(',')){next();if(isOp(')')||isOp('=')||peek().type==='nl'||isOp(']'))break;items.push(starOrExpr())}return{k:'tuple',items}}
  function targetList(){const one=()=>{if(isOp('(')){next();const t=targetList();expect(')');return t}if(isOp('*')){next();return{k:'starred',e:postfix()}}return postfix()};const first=one();if(!isOp(','))return first;const items=[first];while(isOp(',')){next();if(isName('in')||isOp('='))break;items.push(one())}return{k:'tuple',items}}
  function paramList(endOp){const ps=[];while(!isOp(endOp)){let kind='p';if(isOp('*')){next();kind='star';if(isOp(',')){next();continue}}else if(isOp('**')){next();kind='dstar'}const t=next();if(t.type!=='name')err('SyntaxError',`第 ${t.line} 行：参数名写得不对`);
      if(isOp(':')&&endOp===')'){next();expr()}let def=null;if(isOp('=')){next();def=expr()}ps.push({name:t.v,def,kind});if(isOp(','))next();else break}return ps}
  function block(){expect(':');if(peek().type!=='nl'){const s=simpleLine();return s}next();if(peek().type!=='indent')err('IndentationError',`第 ${peek().line} 行：冒号后面的代码要缩进`);next();const body=[];while(peek().type!=='dedent'&&peek().type!=='eof'){if(peek().type==='nl'){next();continue}body.push(...statement())}if(peek().type==='dedent')next();return body}
  function simpleLine(){const out=[simple()];while(isOp(';')){next();if(peek().type==='nl')break;out.push(simple())}if(peek().type!=='nl'&&peek().type!=='eof'&&peek().type!=='dedent')err('SyntaxError',`第 ${peek().line} 行：这一行后面多了「${peek().v??peek().type}」`);if(peek().type==='nl')next();return out}
  function simple(){const line=peek().line;
    if(isName('pass')){next();return{k:'pass',line}}if(isName('break')){next();return{k:'break',line}}if(isName('continue')){next();return{k:'continue',line}}
    if(isName('return')){next();return{k:'return',e:peek().type==='nl'||isOp(';')?null:tupleExpr(),line}}
    if(isName('yield')){next();return{k:'yield',e:peek().type==='nl'?null:tupleExpr(),line}}
    if(isName('global')||isName('nonlocal')){next();while(peek().type==='name'||isOp(','))next();return{k:'pass',line}}
    if(isName('del')){next();return{k:'del',t:tupleExpr(),line}}
    if(isName('raise')){next();const e=peek().type==='nl'?null:expr();return{k:'raise',e,line}}
    if(isName('assert')){next();const c=expr();let m=null;if(isOp(',')){next();m=expr()}return{k:'assert',c,m,line}}
    if(isName('import')||isName('from')){const parts=[];while(peek().type!=='nl'&&!isOp(';')&&peek().type!=='eof'){const t=next();parts.push(t.v)}return{k:'import',parts,line}}
    const lhs=tupleExpr();
    if(isOp('=')){const targets=[lhs];let rhs;for(;;){next();rhs=isName('yield')?(next(),{k:'yieldexp',e:tupleExpr()}):tupleExpr();if(isOp('='))targets.push(rhs);else break}return{k:'assign',targets,value:rhs,line}}
    if(['+=','-=','*=','/=','**=','//=','%=','@='].some(isOp)){const op=next().v.slice(0,-1);return{k:'aug',target:lhs,op,value:tupleExpr(),line}}
    if(isOp(':')&&lhs.k==='name'){next();expr();if(isOp('=')){next();return{k:'assign',targets:[lhs],value:tupleExpr(),line}}return{k:'pass',line}}
    return{k:'expr',e:lhs,line}}
  function statement(){const line=peek().line;
    if(isOp('@')){next();const dec=expr();while(peek().type==='nl')next();const st=statement();const d=st[0];if(!d||(d.k!=='def'&&d.k!=='class'))err('SyntaxError',`第 ${line} 行：装饰器后面要跟 def 或 class`);d.decos=[dec,...(d.decos||[])];return st}
    if(isName('if')){next();const branches=[[expr(),block()]];let orelse=null;for(;;){if(isName('elif')){next();branches.push([expr(),block()])}else if(isName('else')){next();orelse=block();break}else break}return[{k:'if',branches,orelse,line}]}
    if(isName('for')){next();const target=targetList();expectName('in');const iter=tupleExpr();const body=block();let orelse=null;if(isName('else')){next();orelse=block()}return[{k:'for',target,iter,body,orelse,line}]}
    if(isName('while')){next();const c=expr();const body=block();return[{k:'while',c,body,line}]}
    if(isName('def')){next();const name=next().v;expect('(');const params=paramList(')');expect(')');if(isOp('->')){next();expr()}const body=block();return[{k:'def',name,params,body,isGen:hasYield(body),line}]}
    if(isName('class')){next();const name=next().v;let bases=[];if(isOp('(')){next();while(!isOp(')')){bases.push(expr());if(isOp(','))next();else break}expect(')')}const body=block();return[{k:'class',name,bases,body,line}]}
    if(isName('with')){next();const items=[];do{if(isOp(','))next();const e=expr();let as=null;if(isName('as')){next();as=targetList()}items.push({e,as})}while(isOp(','));const body=block();return[{k:'with',items,body,line}]}
    if(isName('try')){next();const body=block();const handlers=[];let fin=null,orelse=null;while(isName('except')){next();let type=null,as=null;if(!isOp(':')){type=expr();if(isName('as')){next();as=next().v}}handlers.push({type,as,body:block()})}if(isName('else')){next();orelse=block()}if(isName('finally')){next();fin=block()}return[{k:'try',body,handlers,orelse,fin,line}]}
    if(peek().type==='indent')err('IndentationError',`第 ${line} 行：这一行不应该缩进`);
    return simpleLine()}
  function hasYield(body){const walk=s=>{if(!s||typeof s!=='object')return false;if(s.k==='yield'||s.k==='yieldexp')return true;if(s.k==='def'||s.k==='class'||s.k==='lambda')return false;return Object.values(s).some(v=>Array.isArray(v)?v.some(walk):walk(v))};return body.some(walk)}
  const stmts=[];while(peek().type!=='eof'){if(peek().type==='nl'||peek().type==='dedent'){next();continue}stmts.push(...statement())}return stmts;
}
function parseExprOnly(src,line){const st=parse(src.trim());if(st.length!==1||st[0].k!=='expr')err('SyntaxError',`第 ${line} 行：f-string 的大括号里只能写表达式`);return st[0].e}

// ---------- 求值 ----------
class Scope{constructor(parent){this.vars=Object.create(null);this.parent=parent}
  get(n){let s=this;while(s){if(n in s.vars)return s.vars[n];s=s.parent}if(n in builtins)return builtins[n];err('NameError',`name '${n}' is not defined`)}
  set(n,v){this.vars[n]=v}}
class Ctl{constructor(kind,v){this.kind=kind;this.v=v}}
let CUR=null;const CTXP={get plots(){return CUR.plots},get fig(){return CUR.fig},set fig(v){CUR.fig=v}};
const D2L=d2lMod(CTXP);
const time=mod('time',{time:fn('time',()=>F((typeof performance!=='undefined'?performance.now():Date.now())/1000)),sleep:fn('sleep',()=>NONE),perf_counter:fn('perf_counter',()=>F((typeof performance!=='undefined'?performance.now():Date.now())/1000))});
const MODULES={torch,numpy,np:numpy,pandas,re:reMod,collections:collectionsMod,os,math,random,time,'os.path':os.attrs.path,'torch.distributions':torch.attrs.distributions,'torch.distributions.multinomial':torch.attrs.distributions.attrs.multinomial,'d2l.torch':D2L,'torch.linalg':torch.attrs.linalg,'matplotlib.pyplot':D2L.attrs.plt,'torch.nn':nnMod,'torch.nn.functional':nnF,'torch.nn.init':nnInit,'torch.nn.utils':nnUtils,'torch.optim':optimMod,'torch.utils':torch.attrs.utils,'torch.utils.data':dataMod,torchvision,'torchvision.datasets':torchvision.attrs.datasets,'torchvision.transforms':torchvision.attrs.transforms,'IPython.display':ipyDisplay,IPython:mod('IPython',{display:ipyDisplay})};
D2L.attrs.set_axes=fn('set_axes',a=>d2lSetAxes(a));D2L.attrs.argmax=fn('argmax',(a,kw)=>argmaxT(asTensor(a[0]),dimArg(a,kw,1),false,1));D2L.attrs.astype=fn('astype',a=>{const T=asTensor(a[0]);return mkT(T.shape.slice(),T.data,dtypeKw({dtype:a[1]},T.dtype),T.np)});
torch.attrs.is_tensor=fn('is_tensor',a=>B(isT(a[0])&&!a[0].np));
function ensurePrelude(){if(D2L.ready)return;D2L.ready=true;const sc=new Scope(null);sc.set('d2l',D2L);const saved=CUR;const ctx={scope:sc,outBuf:'',touched:[],plots:[],figs:[],fig:null,depth:0,steps:0,yields:null};CUR=ctx;try{for(const s of parse(D2L_PRELUDE))exec(s,sc,ctx)}finally{CUR=saved}for(const[k,v]of Object.entries(sc.vars))if(v.t==='func'||v.t==='cls'||(!(k in D2L.attrs)&&v.t!=='mod'&&k!=='d2l'))D2L.attrs[k]=v}
D2L.attrs.load_array=fn('load_array',(a,kw)=>({t:'loader',ds:{t:'tdataset',tensors:iterate(a[0]).map(asTensor)},bs:intOf(a[1]),shuffle:truthy(kwOr(kw,a,'is_train',2,B(true))),drop_last:false}));
function getAttr(o,name){if(isT(o))return tensorAttr(o,name);
  if(name==='__name__'&&(o.t==='type'||o.t==='cls'||o.t==='fn'||o.t==='func')&&o.name)return S(o.t==='type'?o.name.split('.').pop().replace(/^Tensor$/,'Tensor'):o.name);
  switch(o.t){case'mod':{const v=o.attrs[name];if(v!==undefined)return v;err('AttributeError',`module '${o.name}' has no attribute '${name}'`)}
    case'nn':return nnAttr(o,name);case'ax':return axAttr(o,name);case'fig':return DUMMY;case'axarr':if(name==='flatten')return fn('flatten',()=>list(o.fig.axes.slice()));break;
    case'fmnist':if(name==='targets')return mkT([o.n],FM[o.train?'train':'test'].labels,'int64');if(name==='classes')return list(['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot'].map(S));if(name==='train')return B(o.train);break;case'optim':return optAttr(o,name);case'tdataset':if(name==='tensors')return tup(o.tensors);break;
    case'super':{const m=classLookup(o.cls.base,name);if(!m)err('AttributeError',`'super' object has no attribute '${name}'`);return(m.t==='func'||m.method)?{t:'bound',f:m,self:o.self}:m}
    case'df':return dfAttr(o,name);case'series':return seriesAttr(o,name);case'dist':return distAttr(o,name);case'dummy':return DUMMY;
    case'obj':{if(name in o.attrs)return o.attrs[name];const m=classLookup(o.cls,name);if(m){if(m.t==='prop')return callValue(m.f,[o]);if(m.t==='smeth')return m.f;if(m.t==='cmeth')return{t:'bound',f:m.f,self:o.cls};if(m.t==='func'||m.method)return{t:'bound',f:m,self:o};return m}if(isModuleObj(o)&&name==='training')return B(true);err('AttributeError',`'${o.cls.name}' object has no attribute '${name}'`)}
    case'cls':{const m=classLookup(o,name);if(m){if(m.t==='smeth')return m.f;if(m.t==='cmeth')return{t:'bound',f:m.f,self:o};return m}err('AttributeError',`type object '${o.name}' has no attribute '${name}'`)}
    case'gradfn':if(name==='next_functions')return tup(o.g.inputs.filter(isT).map(t=>tup([t.grad_fn?{t:'gradfn',g:t.grad_fn}:NONE,I(0)])));if(name==='name')return fn('name',()=>S(o.g.name));break;
    case'index':if(name==='tolist')return fn('tolist',()=>list(o.items.map(S)));break;
    case'file':{const f=o;const M={write:a=>{const s=str(a[0]);FS.set(f.p,FS.get(f.p)+s);return I(s.length)},read:()=>{const s=FS.get(f.p).slice(f.pos);f.pos=FS.get(f.p).length;return S(s)},readlines:()=>list(iterate(f)),close:()=>NONE};if(M[name])return fn(name,M[name]);break}
    case'iloc':break;case'size':if(name==='numel')return fn('numel',()=>I(prod(o.dims)));break;
    case'dtype':if(name==='is_floating_point')return B(isFloatDt(o.name));break}
  const m=pyMethod(o,name);if(m)return m;err('AttributeError',`'${typeName(o)}' object has no attribute '${name}'`)}
function pyMethod(o,name){const L=o.items;
  if(o.t==='list'){const M={append:a=>{L.push(a[0]);return NONE},extend:a=>{L.push(...iterate(a[0]));return NONE},pop:a=>{if(!L.length)err('IndexError','pop from empty list');let i=a[0]?intOf(a[0]):L.length-1;if(i<0)i+=L.length;return L.splice(i,1)[0]},insert:a=>{L.splice(intOf(a[0]),0,a[1]);return NONE},
      index:a=>{const i=L.findIndex(x=>pyEq(x,a[0]));if(i<0)err('ValueError',`${repr(a[0])} is not in list`);return I(i)},count:a=>I(L.filter(x=>pyEq(x,a[0])).length),reverse:()=>{L.reverse();return NONE},clear:()=>{L.length=0;return NONE},copy:()=>list(L.slice()),
      sort:(a,kw)=>{const s=builtins.sorted.call([o],kw).items;L.splice(0,L.length,...s);return NONE}};if(M[name])return fn(name,M[name])}
  if(o.t==='tuple'){const M={index:a=>I(L.findIndex(x=>pyEq(x,a[0]))),count:a=>I(L.filter(x=>pyEq(x,a[0])).length)};if(M[name])return fn(name,M[name])}
  if(o.t==='dict'&&o.counter){if(name==='most_common')return fn('most_common',a=>{const r=[...o.m.values()].sort((x,y)=>y[1].v-x[1].v);return list((a[0]&&a[0].t!=='none'?r.slice(0,intOf(a[0])):r).map(([k,v])=>tup([k,v])))});if(name==='update')return fn('update',a=>{if(a[0].t==='dict')for(const[k,v]of a[0].m.values()){const e=o.m.get(numKey(k));o.m.set(numKey(k),[k,I((e?e[1].v:0)+v.v)])}else counterAdd(o,iterate(a[0]));return NONE});if(name==='total')return fn('total',()=>I([...o.m.values()].reduce((s,e)=>s+e[1].v,0)))}
  if(o.t==='dict'){const vw=(x,n)=>{x.view=(o.odict?'odict_':'dict_')+n;return x};const M={items:()=>vw(list([...o.m.values()].map(([k,v])=>tup([k,v]))),'items'),keys:()=>vw(list([...o.m.values()].map(e=>e[0])),'keys'),values:()=>vw(list([...o.m.values()].map(e=>e[1])),'values'),get:a=>{const e=o.m.get(numKey(a[0]));return e?e[1]:(a[1]||NONE)},update:a=>{for(const[k,v]of a[0].m.values())o.m.set(numKey(k),[k,v]);return NONE},pop:a=>{const k=numKey(a[0]),e=o.m.get(k);if(!e){if(a[1])return a[1];err('KeyError',repr(a[0]))}o.m.delete(k);return e[1]},setdefault:a=>{const k=numKey(a[0]);if(!o.m.has(k))o.m.set(k,[a[0],a[1]||NONE]);return o.m.get(k)[1]}};if(M[name])return fn(name,M[name])}
  if(o.t==='str'){const s=o.v;const M={join:a=>S(iterate(a[0]).map(x=>{if(x.t!=='str')err('TypeError',`sequence item: expected str instance, ${typeName(x)} found`);return x.v}).join(s)),split:a=>{const sep=a[0]&&a[0].t!=='none'?a[0].v:null;return list((sep==null?s.trim().split(/\s+/).filter(Boolean):s.split(sep)).map(S))},strip:a=>S(a[0]?s.replace(new RegExp(`^[${a[0].v}]+|[${a[0].v}]+$`,'g'),''):s.trim()),
      upper:()=>S(s.toUpperCase()),lower:()=>S(s.toLowerCase()),replace:a=>S(s.split(a[0].v).join(a[1].v)),startswith:a=>B(s.startsWith(a[0].v)),endswith:a=>B(s.endsWith(a[0].v)),format:(a,kw)=>{let i=0;return S(s.replace(/\{([^{}]*)\}/g,(m0,inner)=>{const[k,spec]=inner.split(':');const v=k===''?a[i++]:/^\d+$/.test(k)?a[+k]:kw[k];return formatSpec(v,spec||'')}))},
      count:a=>I(s.split(a[0].v).length-1),find:a=>I(s.indexOf(a[0].v)),isdigit:()=>B(/^\d+$/.test(s)),rstrip:()=>S(s.replace(/\s+$/,'')),lstrip:()=>S(s.replace(/^\s+/,'')),title:()=>S(s.replace(/\b\w/g,c=>c.toUpperCase())),ljust:a=>S(s.padEnd(intOf(a[0]))),rjust:a=>S(s.padStart(intOf(a[0]))),zfill:a=>S(s.padStart(intOf(a[0]),'0'))};if(M[name])return fn(name,M[name])}
  if(o.t==='int'||o.t==='float'){if(name==='is_integer')return fn('is_integer',()=>B(Number.isInteger(o.v)));if(name==='real')return o}
  return null}
function classLookup(cls,name){let c=cls;while(c){if(name in c.attrs)return c.attrs[name];c=c.base}return null}
function callValue(f,args,kw,ctx){ctx=ctx||CUR;
  if(f.t==='fn')return f.call(args,kw,ctx);
  if(f.t==='bound')return f.f.t==='fn'?f.f.call([f.self,...args],kw,ctx):callFunc(f.f,[f.self,...args],kw,ctx);
  if(f.t==='nn')return nnForward(f,args,ctx);
  if(f.t==='func')return callFunc(f,args,kw,ctx);
  if(f.t==='cls'){const obj={t:'obj',cls:f,attrs:Object.create(null)};const init=classLookup(f,'__init__');if(init)callFunc(init,[obj,...args],kw,ctx);return obj}
  if(f.t==='obj'){const c=classLookup(f.cls,'__call__')||classLookup(f.cls,'forward');if(c)return c.t==='fn'?c.call([f,...args],kw,ctx):callFunc(c,[f,...args],kw,ctx)}
  if(f.t==='type'&&builtins[f.name])return builtins[f.name].call(args,kw,ctx);
  err('TypeError',`'${typeName(f)}' object is not callable`)}
function callFunc(f,args,kw,ctx){if(++ctx.depth>200)err('RecursionError','maximum recursion depth exceeded');const sc=new Scope(f.closure);if(f.owner){sc.set('__class__',f.owner);if(args.length)sc.set('__self__',args[0])}const prevScope=ctx.scope;ctx.scope=sc;let ai=0;const extraKw={...kw};
  for(const p of f.params){if(p.kind==='star'){sc.set(p.name,tup(args.slice(ai)));ai=args.length;continue}if(p.kind==='dstar'){sc.set(p.name,mkDict(Object.entries(extraKw).map(([k,v])=>[S(k),v])));for(const k in extraKw)delete extraKw[k];continue}
    if(ai<args.length){if(p.name in extraKw)err('TypeError',`${f.name}() got multiple values for argument '${p.name}'`);sc.set(p.name,args[ai++])}else if(p.name in extraKw){sc.set(p.name,extraKw[p.name]);delete extraKw[p.name]}else if(p.def)sc.set(p.name,p.defVal);else err('TypeError',`${f.name}() missing 1 required positional argument: '${p.name}'`)}
  if(ai<args.length)err('TypeError',`${f.name}() takes ${f.params.filter(p=>p.kind==='p').length} positional arguments but ${args.length} were given`);
  const left=Object.keys(extraKw);if(left.length)err('TypeError',`${f.name}() got an unexpected keyword argument '${left[0]}'`);
  try{if(f.expr){return evalE(f.expr,sc,ctx)}
    if(f.isGen){const ys=[];const saved=ctx.yields;ctx.yields=ys;try{execBlock(f.body,sc,ctx)}catch(e){if(!(e instanceof Ctl&&e.kind==='return'))throw e}finally{ctx.yields=saved}return{t:'iter',items:ys,pos:0}}
    try{execBlock(f.body,sc,ctx)}catch(e){if(e instanceof Ctl&&e.kind==='return')return e.v;throw e}return NONE}finally{ctx.depth--;ctx.scope=prevScope}}
function subItems(items,env,ctx){return items.map(it=>{if(it.e&&it.e.k){}if(it.ellipsis)return it;if(it.slice)return{slice:it.slice.map(x=>{if(x==null)return null;const v=evalE(x,env,ctx);return v.t==='none'?null:intOf(v)})};
  const v=evalE(it.e,env,ctx);if(v.t==='none')return{none:true};if(isT(v)){if(v.dtype==='bool')return{bool:v};if(v.shape.length===0)return{v:v.data[0]};return{tensor:v}}if(v.t==='list')return{tensor:mkT([v.items.length],v.items.map(intOf),'int64')};if(v.t==='range')return{tensor:mkT([pyLen(v)],iterate(v).map(intOf),'int64')};return{v:intOf(v),raw:v}})}
function getSub(o,e,env,ctx){if(!env)env=new Scope(null);
  if(o.t==='obj'){const g=classLookup(o.cls,'__getitem__');if(!g)err('TypeError',`'${o.cls.name}' object is not subscriptable`);const ks=e.items.map(it=>it.slice?{t:'slice',v:it.slice.map(x=>x==null?NONE:evalE(x,env,ctx))}:evalE(it.e,env,ctx));return callValue({t:'bound',f:g,self:o},[ks.length===1&&!e.trailingComma?ks[0]:tup(ks)],{},ctx)}
  if(o.t==='dict'){const k=evalE(e.items[0].e,env,ctx);const ent=o.m.get(numKey(k));if(!ent){if(o.counter)return I(0);if(o.dflt){const v=callValue(o.dflt,[],{},ctx);o.m.set(numKey(k),[k,v]);return v}err('KeyError',repr(k))}return ent[1]}
  if(o.t==='df'){const k=evalE(e.items[0].e,env,ctx);if(k.t==='str'){if(!o.cols.includes(k.v))err('KeyError',`'${k.v}'`);return colSeries(o,k.v)}if(k.t==='list')return dfSelectCols(o,k.items.map(str));if(k.t==='series'&&k.dtype==='bool')return dfSelectRows(o,k.values.map((v,i)=>v?i:-1).filter(i=>i>=0));err('KeyError',repr(k))}
  if(o.t==='series'){const k=evalE(e.items[0].e,env,ctx);const i=o.index.findIndex(x=>String(x)===str(k));if(i<0)err('KeyError',repr(k));return pyVal(o.values[i],o.dtype)}
  if(o.t==='iloc')return ilocGet(o.df,subItems(e.items,env,ctx));
  if(o.t==='nn'&&o.kind==='Sequential'){const it=subItems(e.items,env,ctx)[0];if(it.slice){const r=getSub(list(o.children.map(c=>c[1])),e,env,ctx);return nnObj('Sequential',{},{},r.items.map((v,i)=>[String(i),v]))}let i=it.v;if(i<0)i+=o.children.length;if(i<0||i>=o.children.length)err('IndexError',`index ${it.v} is out of range`);return o.children[i][1]}
  if(o.t==='fmnist')return fmGet(o,intOf(evalE(e.items[0].e,env,ctx)));
  if(o.t==='axarr'){const its=subItems(e.items,env,ctx);const i=its.length>1?its[0].v*o.cols+its[1].v:its[0].v;return o.fig.axes[i<0?i+o.fig.axes.length:i]}
  if(o.t==='tdataset'){const i=intOf(evalE(e.items[0].e,env,ctx));return tup(o.tensors.map(t=>getItem(t,[{v:i}])))}
  if(o.t==='dummy')return DUMMY;
  const items=subItems(e.items,env,ctx);if(isT(o))return getItem(o,items);
  if(o.t==='tuple'||o.t==='list'||o.t==='size'||o.t==='str'||o.t==='range'){const arr=o.t==='str'?[...o.v].map(S):o.t==='size'?o.dims.map(I):o.t==='range'?iterate(o):o.items;const it=items[0];
    if(it.slice){let[a,b,c]=it.slice;c=c==null?1:c;const n=arr.length;const nrm=(x,d)=>x==null?d:x<0?Math.max(x+n,c<0?-1:0):Math.min(x,c<0?n-1:n);a=nrm(a,c<0?n-1:0);b=nrm(b,c<0?-1:n);const r=[];if(c>0)for(let i=a;i<b;i+=c)r.push(arr[i]);else for(let i=a;i>b;i+=c)r.push(arr[i]);
      if(o.t==='str')return S(r.map(x=>x.v).join(''));if(o.t==='size')return{t:'size',dims:r.map(x=>x.v)};return{t:o.t==='range'?'list':o.t,items:r}}
    let i=it.v;if(i<0)i+=arr.length;if(i<0||i>=arr.length)err('IndexError',`${typeName(o)} index out of range`);return arr[i]}
  if(o.t==='obj'){const g=classLookup(o.cls,'__getitem__');if(g)return callFunc(g,[o,evalE(e.items[0].e,env,ctx)],{},ctx)}
  err('TypeError',`'${typeName(o)}' object is not subscriptable`)}
function evalCall(e,env,ctx){const f=evalE(e.fn,env,ctx);const args=[];for(const a of e.args){if(a.star)args.push(...iterate(evalE(a.star,env,ctx)));else if(a.k==='gen')args.push(evalE(a,env,ctx));else args.push(evalE(a,env,ctx))}
  const kw={};for(const k of e.kw){if(k.dstar){const d=evalE(k.dstar,env,ctx);for(const[kk,v]of d.m.values())kw[str(kk)]=v}else kw[k.k]=evalE(k.v,env,ctx)}return callValue(f,args,kw,ctx)}
function compRun(c,env,ctx,emit){const sc=new Scope(env);const rec=i=>{if(i===c.clauses.length){emit(sc);return}const cl=c.clauses[i];if(cl.k==='for'){for(const v of iterate(evalE(cl.iter,sc,ctx))){assignTo(cl.target,v,sc,ctx);rec(i+1)}}else if(truthy(evalE(cl.cond,sc,ctx)))rec(i+1)};rec(0)}
function contains(c,x){if(c.t==='dict')return c.m.has(numKey(x));if(c.t==='str')return c.v.includes(x.v);if(c.t==='df')return c.cols.includes(str(x));if(isT(c))return c.data.some(v=>v===numOf(x));return iterate(c).some(y=>pyEq(x,y))}
function evalE(e,env,ctx){switch(e.k){
  case'num':return e.fl?F(e.v):I(e.v);case'str':return S(e.v);case'const':return e.v;
  case'fstr':return S(e.parts.map(p=>{if(typeof p==='string')return p;let v=evalE(p.ast,env,ctx);let s=p.conv==='r'?repr(v):p.spec?formatSpec(v,p.spec):str(v);return p.dbg?p.dbg+(p.spec?s:repr(v)):s}).join(''));
  case'name':return env.get(e.v);
  case'tuple':{const items=[];for(const x of e.items){if(x.k==='starred')items.push(...iterate(evalE(x.e,env,ctx)));else items.push(evalE(x,env,ctx))}return tup(items)}
  case'list':{const items=[];for(const x of e.items){if(x.k==='starred')items.push(...iterate(evalE(x.e,env,ctx)));else items.push(evalE(x,env,ctx))}return list(items)}
  case'setlit':return list(e.items.map(x=>evalE(x,env,ctx)));
  case'dict':return mkDict(e.pairs.map(([k,v])=>[evalE(k,env,ctx),evalE(v,env,ctx)]));
  case'listcomp':{const r=[];compRun(e,env,ctx,sc=>r.push(evalE(e.elt,sc,ctx)));return list(r)}
  case'gen':{const r=[];compRun(e,env,ctx,sc=>r.push(evalE(e.elt,sc,ctx)));return{t:'iter',items:r,pos:0}}
  case'dictcomp':{const r=[];compRun(e,env,ctx,sc=>r.push([evalE(e.key,sc,ctx),evalE(e.val,sc,ctx)]));return mkDict(r)}
  case'neg':{const v=evalE(e.a,env,ctx);if(isT(v))return unary(v,'neg');if(v.t==='int'||v.t==='bool')return I(-pyNum(v));if(v.t==='float')return F(-v.v);if(v.t==='series')return seriesBinop('*',v,I(-1));err('TypeError',`bad operand type for unary -: '${typeName(v)}'`)}
  case'inv':{const v=evalE(e.a,env,ctx);if(isT(v)&&v.dtype==='bool')return mkT(v.shape.slice(),Array.from(v.data,x=>!x),'bool',v.np);return I(~intOf(v))}
  case'not':return B(!truthy(evalE(e.a,env,ctx)));
  case'and':{const a=evalE(e.a,env,ctx);return truthy(a)?evalE(e.b,env,ctx):a}case'or':{const a=evalE(e.a,env,ctx);return truthy(a)?a:evalE(e.b,env,ctx)}
  case'ifexp':return truthy(evalE(e.c,env,ctx))?evalE(e.a,env,ctx):evalE(e.b,env,ctx);
  case'lambda':return{t:'func',name:'<lambda>',params:e.params.map(p=>({...p,defVal:p.def?evalE(p.def,env,ctx):null})),expr:e.body,closure:env,id:newId()};
  case'bin':{const a=evalE(e.a,env,ctx),b=evalE(e.b,env,ctx);if(e.op==='@')return matmul(asTensor(a),asTensor(b));
    if(a.t==='obj'){const m=classLookup(a.cls,{'+':'__add__','-':'__sub__','*':'__mul__','/':'__truediv__'}[e.op]);if(m)return callFunc(m,[a,b],{},ctx)}return binop(e.op,a,b)}
  case'cmp':{let left=evalE(e.first,env,ctx),res=null;for(let i=0;i<e.ops.length;i++){const right=evalE(e.rest[i],env,ctx),op=e.ops[i];let r;
      if(op==='in')r=B(contains(right,left));else if(op==='not in')r=B(!contains(right,left));else if(op==='is')r=B(left===right||(left.t==='none'&&right.t==='none')||(left.t==='bool'&&right.t==='bool'&&left.v===right.v));else if(op==='is not')r=B(!(left===right||(left.t==='none'&&right.t==='none')||(left.t==='bool'&&right.t==='bool'&&left.v===right.v)));else r=binop(op,left,right);
      if(e.ops.length===1)return r;if(!truthy(r))return r;res=r;left=right}return res}
  case'attr':return getAttr(evalE(e.obj,env,ctx),e.name);
  case'call':return evalCall(e,env,ctx);
  case'sub':return getSub(evalE(e.obj,env,ctx),e,env,ctx);
  case'starred':err('SyntaxError','星号 * 只能用在函数调用或解包里');
  case'yieldexp':ctx.yields.push(evalE(e.e,env,ctx));return NONE}
  err('SyntaxError','不支持的写法')}
function assignTo(t,val,env,ctx){
  if(t.k==='name'){env.set(t.v,val);return}
  if(t.k==='tuple'||t.k==='list'){if(val.t==='dummy'){t.items.forEach(x=>assignTo(x,DUMMY,env,ctx));return}const vs=iterate(val);const si=t.items.findIndex(x=>x.k==='starred');if(si<0){if(vs.length!==t.items.length)err('ValueError',vs.length>t.items.length?`too many values to unpack (expected ${t.items.length})`:`not enough values to unpack (expected ${t.items.length}, got ${vs.length})`);t.items.forEach((x,i)=>assignTo(x,vs[i],env,ctx));return}
    const after=t.items.length-si-1;t.items.slice(0,si).forEach((x,i)=>assignTo(x,vs[i],env,ctx));assignTo(t.items[si].e,list(vs.slice(si,vs.length-after)),env,ctx);t.items.slice(si+1).forEach((x,i)=>assignTo(x,vs[vs.length-after+i],env,ctx));return}
  if(t.k==='sub'){const o=evalE(t.obj,env,ctx);
    if(o.t==='dict'){const k=evalE(t.items[0].e,env,ctx);o.m.set(numKey(k),[k,val]);return}if(o.t==='list'){const i0=intOf(evalE(t.items[0].e,env,ctx));const i=i0<0?i0+o.items.length:i0;if(i<0||i>=o.items.length)err('IndexError','list assignment index out of range');o.items[i]=val;return}
    if(o.t==='dummy')return;if(o.t==='df'){const k=str(evalE(t.items[0].e,env,ctx));const vals=val.t==='series'?val.values:o.index.map(()=>pyNum(val)??str(val));if(!o.cols.includes(k))o.cols.push(k);o.data[k]=vals.slice();o.dtype[k]=val.t==='series'?val.dtype:val.t==='str'?'str':val.t==='float'?'float64':'int64';return}
    if(!isT(o))err('TypeError',`'${typeName(o)}' object does not support item assignment`);setItem(o,subItems(t.items,env,ctx),val);ctx.touched.push(o);return}
  if(t.k==='attr'){const o=evalE(t.obj,env,ctx);if(o.t==='obj'){o.attrs[t.name]=val;return}if(o.t==='dummy')return;
    if(isT(o)){if(t.name==='requires_grad'){if(truthy(val)&&!isFloatDt(o.dtype))err('RuntimeError','only Tensors of floating point dtype can require gradients');o.requires_grad=truthy(val);return}if(t.name==='grad'){o.grad=val.t==='none'?null:val;return}if(t.name==='data'){o.data=Float64Array.from(val.data);return}}
    err('AttributeError',`不能给 ${typeName(o)} 设置属性 ${t.name}`)}
  if(t.k==='starred')return assignTo(t.e,list(iterate(val)),env,ctx);
  err('SyntaxError','这个东西不能放在等号左边')}
function doImport(s,env){const p=s.parts;
  if(p.join(' ').includes('d2l'))ensurePrelude();
  if(p[0]==='import'){const names=[];let i=1;while(i<p.length){let name='';while(i<p.length&&p[i]!==','&&p[i]!=='as'){name+=p[i];i++}let as=null;if(p[i]==='as'){as=p[i+1];i+=2}if(p[i]===',')i++;names.push([name,as])}
    for(const[name,as]of names){const m=MODULES[name];if(!m){if(['matplotlib','collections','hashlib','zipfile','tarfile','requests','sys'].includes(name.split('.')[0])){env.set(as||name.split('.')[0],DUMMY);continue}err('ModuleNotFoundError',`No module named '${name}'（本页的简化环境里没有这个包）`)}
      if(as)env.set(as,m);else{const top=name.split('.')[0];env.set(top,MODULES[top]||m)}}return}
  // from X import a as b, c
  const ii=p.indexOf('import');const modName=p.slice(1,ii).join('');const rest=p.slice(ii+1).filter(x=>x!=='('&&x!==')');let i=0;
  while(i<rest.length){const name=rest[i];let as=null;if(rest[i+1]==='as'){as=rest[i+2];i+=3}else i++;if(rest[i]===',')i++;
    const full=modName+'.'+name;let v=MODULES[full];if(!v){const m=MODULES[modName];if(m&&m.attrs[name]!==undefined)v=m.attrs[name];else if(m||['matplotlib_inline','IPython','matplotlib','collections','typing'].includes(modName.split('.')[0]))v=DUMMY;else err('ModuleNotFoundError',`No module named '${modName}'（本页的简化环境里没有这个包）`)}
    env.set(as||name,v)}}
function execBlock(stmts,env,ctx){for(const s of stmts)exec(s,env,ctx)}
function exec(s,env,ctx){ctx.line=s.line;if(++ctx.steps>3e6)err('TimeoutError','代码运行太久了（可能是死循环），已停止');
  switch(s.k){case'expr':{const v=evalE(s.e,env,ctx);ctx.last=v;return}
    case'assign':{const v=evalE(s.value,env,ctx);for(const t of s.targets)assignTo(t,v,env,ctx);ctx.last=undefined;return}
    case'aug':{const t=s.target,cur=evalE(t,env,ctx),v=evalE(s.value,env,ctx);ctx.last=undefined;
      if(t.k==='sub'&&isT(cur)){const o=evalE(t.obj,env,ctx);if(isT(o)){checkInplace(o);setItem(o,subItems(t.items,env,ctx),binop(s.op,cur,v));ctx.touched.push(o);return}}
      if(isT(cur)){checkInplace(cur);const r=s.op==='@'?matmul(cur,asTensor(v)):binop(s.op,cur,v);if(r.shape.join()!==cur.shape.join())err('RuntimeError',`output with shape [${cur.shape.join(', ')}] doesn't match the broadcast shape [${r.shape.join(', ')}]`);
        if(RANK[r.dtype]>RANK[cur.dtype])err('RuntimeError',`result type ${r.dtype==='float32'?'Float':r.dtype==='float64'?'Double':'Long'} can't be cast to the desired output type ${cur.dtype==='int64'?'Long':cur.dtype==='float32'?'Float':'Bool'}`);
        for(let i=0;i<r.data.length;i++)cur.data[i]=castVal(r.data[i],cur.dtype);if(r.grad_fn&&!cur.grad_fn){/* 非叶子原地更新：保持简单 */}cur.meta={kind:'assign',mask:new Uint8Array(cur.data.length).fill(1)};ctx.touched.push(cur);return}
      if(cur.t==='list'&&s.op==='+'){cur.items.push(...iterate(v));return}
      assignTo(t,binop(s.op,cur,v),env,ctx);return}
    case'import':doImport(s,env);ctx.last=undefined;return;
    case'pass':ctx.last=undefined;return;case'break':throw new Ctl('break');case'continue':throw new Ctl('continue');
    case'return':throw new Ctl('return',s.e?evalE(s.e,env,ctx):NONE);
    case'yield':if(!ctx.yields)err('SyntaxError',"'yield' outside function");ctx.yields.push(s.e?evalE(s.e,env,ctx):NONE);return;
    case'raise':{const v=s.e?evalE(s.e,env,ctx):null;err(v&&v.excType?v.excType:'Exception',v&&v.msg!=null?v.msg:v?str(v):'')}
    case'assert':if(!truthy(evalE(s.c,env,ctx)))err('AssertionError',s.m?str(evalE(s.m,env,ctx)):'');return;
    case'del':{const t=s.t;if(t.k==='name'){let sc=env;while(sc&&!(t.v in sc.vars))sc=sc.parent;if(!sc)err('NameError',`name '${t.v}' is not defined`);delete sc.vars[t.v]}return}
    case'if':ctx.last=undefined;for(const[c,b]of s.branches){if(truthy(evalE(c,env,ctx))){execBlock(b,env,ctx);ctx.last=undefined;return}}if(s.orelse)execBlock(s.orelse,env,ctx);ctx.last=undefined;return;
    case'for':{const src=evalE(s.iter,env,ctx);const items=src.t==='loader'?loaderGen(src,ctx):src.t==='iter'&&src.gen?src.gen:iterate(src);let broke=false;for(const v of items){assignTo(s.target,v,env,ctx);try{execBlock(s.body,env,ctx)}catch(e){if(e instanceof Ctl&&e.kind==='break'){broke=true;break}if(e instanceof Ctl&&e.kind==='continue')continue;throw e}}if(!broke&&s.orelse)execBlock(s.orelse,env,ctx);ctx.last=undefined;return}
    case'while':{while(truthy(evalE(s.c,env,ctx))){if(++ctx.steps>3e6)err('TimeoutError','代码运行太久了（可能是死循环），已停止');try{execBlock(s.body,env,ctx)}catch(e){if(e instanceof Ctl&&e.kind==='break')break;if(e instanceof Ctl&&e.kind==='continue')continue;throw e}}ctx.last=undefined;return}
    case'def':{let f={t:'func',name:s.name,params:s.params.map(p=>({...p,defVal:p.def?evalE(p.def,env,ctx):null})),body:s.body,isGen:s.isGen,closure:env,id:newId()};if(s.decos)for(const d of s.decos.slice().reverse())f=callValue(evalE(d,env,ctx),[f],{},ctx);env.set(s.name,f);ctx.last=undefined;return}
    case'class':{let base=null;if(s.bases.length){const b=evalE(s.bases[0],env,ctx);if(b.t==='cls')base=b}const cls={t:'cls',name:s.name,attrs:Object.create(null),base};const sc=new Scope(env);execBlock(s.body,sc,ctx);Object.assign(cls.attrs,sc.vars);for(const v of Object.values(cls.attrs))if(v&&v.t==='func')v.owner=cls;env.set(s.name,cls);ctx.last=undefined;return}
    case'with':{const exits=[];try{for(const it of s.items){const v=evalE(it.e,env,ctx);let entered=v;if(v.t==='ctx'){exits.push(v.enter())}else if(v.t==='file'){exits.push(()=>{})}else if(v.t==='dummy'){}else err('TypeError',`'${typeName(v)}' object does not support the context manager protocol`);if(it.as)assignTo(it.as,entered,env,ctx)}
      execBlock(s.body,env,ctx)}finally{for(const x of exits.reverse())x()}ctx.last=undefined;return}
    case'try':{try{execBlock(s.body,env,ctx);if(s.orelse)execBlock(s.orelse,env,ctx)}catch(e){if(!(e instanceof PyErr))throw e;const h=s.handlers.find(h=>!h.type||(h.type.k==='name'&&(h.type.v==='Exception'||h.type.v===e.pyType))||(h.type.k==='tuple'&&h.type.items.some(x=>x.v===e.pyType||x.v==='Exception')));if(!h)throw e;if(h.as)env.set(h.as,S(e.message));execBlock(h.body,env,ctx)}finally{if(s.fin)execBlock(s.fin,env,ctx)}ctx.last=undefined;return}}
  err('SyntaxError','不支持的语句')}
for(const n of['Exception','ValueError','TypeError','RuntimeError','KeyError','IndexError','StopIteration','ZeroDivisionError','AssertionError','NotImplementedError'])builtins[n]=fn(n,a=>({t:'exc',excType:n,msg:a[0]?str(a[0]):''}));
function run(src,env){warnings=[];const ctx={scope:env,figs:[],outBuf:'',touched:[],plots:[],fig:null,depth:0,steps:0,yields:null,last:undefined};CUR=ctx;
  for(const v of Object.values(env.vars))if(isT(v))v.meta=null;
  const stmts=parse(src);
  try{for(const s of stmts){exec(s,env,ctx)}}catch(e){if(e instanceof Ctl)err('SyntaxError',`'${e.kind}' 只能写在${e.kind==='return'?'函数':'循环'}里`);e.partialOut=ctx.outBuf;for(const f of ctx.figs)ctx.plots.push(figSpec(f));e.plots=ctx.plots;throw e}
  if(warnings.length){ctx.outBuf+=warnings.join('\n')+'\n';warnings=[]}
  for(const f of ctx.figs)ctx.plots.push(figSpec(f));
  const last=stmts.length&&stmts[stmts.length-1].k==='expr'&&!/;\s*(#[^\n]*)?$/.test(src.trimEnd())?ctx.last:undefined;
  return{out:ctx.outBuf,value:last&&last.t!=='none'?last:undefined,touched:ctx.touched,plots:ctx.plots}}
function runCell(src,env){gradEnabled=true;try{const r=run(src,env);const val=r.value!==undefined?repr(r.value):'';const text=(r.out+(val?val:'')).replace(/\n+$/,'');return{ok:true,...r,text}}
  catch(e){gradEnabled=true;if(e instanceof PyErr)return{ok:false,errType:e.pyType,text:(e.partialOut||'')+`${e.pyType}: ${e.message}`,plots:e.plots||[]};return{ok:false,errType:'InternalError',text:`本页的迷你解释器出错了：${e.message}`,plots:[]}}}
return{setFashion,setFashionArrays,runCell,repr,isT,Tensor,shapeOf:t=>t.shape,newEnv:()=>new Scope(null),strides,FS,setStaticFile,resetFS:()=>{FS.clear();DIRS.clear();DIRS.add('.');DIRS.add('..');addStatic()}}})();
