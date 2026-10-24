
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
