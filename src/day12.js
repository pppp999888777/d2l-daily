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
