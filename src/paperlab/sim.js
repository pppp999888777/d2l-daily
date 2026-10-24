// 论文的 BKT 风格模拟器（experiment.py 第 810 行 rollout）的 JavaScript 版
function mulberry(seed){let s=seed>>>0;return()=>{s=s+0x6D2B79F5>>>0;let t=Math.imul(s^s>>>15,1|s);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
const sig=x=>1/(1+Math.exp(-x)),lgt=p=>Math.log(p/(1-p));
function makeSim(par){const K=par.p0.length;return{K,p0:par.p0,l:par.learn,g:par.guess,s:par.slip,fs:par.forget_per_step}}
// policy(state) -> 返回每个学生选的知识点；state 提供历史计数和真实掌握（只给“理想模型”用）
function rollout(sim,policy,n,T,seed,idle=20){const K=sim.K,u=mulberry(seed),gauss=()=>{let a=0;while(!a)a=u();return Math.sqrt(-2*Math.log(a))*Math.cos(2*Math.PI*u())};
  const L=[],leff=[],S=[],N=[],streak=[],first=[];let pre=0,post=0,selm=0,distinct=0,mg=0;const perf=Lr=>Lr.reduce((s,v,k)=>s+(v?1-sim.s[k]:sim.g[k]),0)/K;
  const traj=[];
  for(let i=0;i<n;i++){const th=0.7*gauss();L.push(sim.p0.map(p=>u()<sig(lgt(p)+th)));leff.push(sim.l.map(p=>sig(lgt(p)+0.5*th)));S.push(new Array(K).fill(0));N.push(new Array(K).fill(0));streak.push(new Array(K).fill(0));first.push(new Array(K).fill(T+1))}
  const preArr=L.map(perf),preFrac=L.map(r=>r.filter(Boolean).length/K);
  for(let i=0;i<n;i++){const used=new Set();const rec=[];for(let t=0;t<T;t++){const a=policy({sim,S:S[i],N:N[i],L:L[i],t,u,K});used.add(a);const Lk=L[i][a],p=Lk?1-sim.s[a]:sim.g[a],y=u()<p;
      const learn=!Lk&&u()<leff[i][a];if(Lk)selm++;for(let k=0;k<K;k++)if(k!==a&&L[i][k]&&u()<sim.fs[k])L[i][k]=false;if(learn)L[i][a]=true;
      S[i][a]+=y?1:0;N[i][a]++;streak[i][a]=y?streak[i][a]+1:0;if(streak[i][a]>=3&&first[i][a]>T)first[i][a]=t+1;if(i<1)rec.push([a,y,Lk,learn])}
    distinct+=used.size;if(i<1)traj.push(rec)}
  const postArr=L.map(perf),postFrac=L.map(r=>r.filter(Boolean).length/K);let nlg=0,stm=0;for(let i=0;i<n;i++){nlg+=(postArr[i]-preArr[i])/Math.max(1-preArr[i],1e-3);stm+=first[i].reduce((a,b)=>a+b,0)/K;mg+=postFrac[i]-preFrac[i]}
  return{nlg:nlg/n,stm:stm/n,fracMastered:selm/(n*T),distinct:distinct/n,masteryGain:mg/n,traj:traj[0]}}
// 策略
const POL={
  random:()=>st=>Math.min(st.K-1,Math.floor(st.u()*st.K)),
  logging:()=>st=>{const{S,N,K}=st,acc=S.map((s,k)=>(s+1)/(N[k]+2)),sc=acc.map(a=>-((a-0.7)**2)/0.02),m=Math.max(...sc),e=sc.map(v=>Math.exp(v-m)),z=e.reduce((a,b)=>a+b,0);const p=e.map(v=>0.5/K+0.5*v/z);let c=st.u(),k=0;while(k<K-1&&(c-=p[k])>=0)k++;return k},
  // 论文的 ModelPolicy：第 1 步随机；之后选“预测答对概率”最接近 c 的知识点。经验估计和 EmpAcc 一样：(答对数 + 2·先验) / (做题数 + 2)，先验取这个知识点的平均正确率
  target:(c,est)=>st=>{const{S,N,K,sim,L,t}=st;if(t===0)return Math.min(K-1,Math.floor(st.u()*K));let best=0,bd=Infinity;for(let k=0;k<K;k++){const m=sim.p0[k]*(1-sim.s[k])+(1-sim.p0[k])*sim.g[k],p=est==='oracle'?(L[k]?1-sim.s[k]:sim.g[k]):(S[k]+2*m)/(N[k]+2);const d=Math.abs(p-c)+1e-9*k;if(d<bd){bd=d;best=k}}return best}};
if(typeof module!=='undefined')module.exports={makeSim,rollout,POL};
