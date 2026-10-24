
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
