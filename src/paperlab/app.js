const D=JSON.parse(document.getElementById('D').textContent);
const $=s=>document.querySelector(s),$$=s=>[...document.querySelectorAll(s)];
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const DSN={assist2009:'ASSISTments 2009',ednet:'EdNet',algebra2005:'Algebra 2005',junyi:'Junyi'};
const DSL=['assist2009','ednet','algebra2005','junyi'];
const AG='Safe-DFKT-AG',DEP='Safe-DFKT w/o policy head (two-stage deploy)';
const COURSE='https://pppp999888777.github.io/d2l-daily/daily.html#day';
// 方法的中文名、分组和说明
const M={
 'LR':{zh:'LR',g:'kt',d:'逻辑回归：每个知识点一个偏置，加上做题、答对、答错次数（取对数）的特征。',day:6},
 'PFA':{zh:'PFA',g:'kt',d:'表现因素分析：每个知识点的难度，加上答对次数和答错次数各自的权重。',day:6},
 'Heuristic-0.7':{zh:'Heuristic-0.7',g:'ref',d:'不用 KT 模型：每个知识点的平滑正确率，选最接近 0.7 的。',day:null},
 'Random':{zh:'Random',g:'ref',d:'均匀随机选知识点。没有预测，所以没有 AUC。教学上的参照点。',day:null},
 'DKT':{zh:'DKT',g:'kt',d:'LSTM（d = 64）。两阶段：先用 BCE 训练，再选预测答对概率最接近 0.7 的知识点。',day:14},
 'Uncertainty-0.5':{zh:'Uncertainty-0.5',g:'ref',d:'和 DKT 完全一样的预测，只是改选最接近 0.5 的（最没把握的）知识点。',day:14},
 'DKVMN':{zh:'DKVMN',g:'kt',d:'动态键值记忆网络，20 个记忆槽，读记忆用点积注意力。',day:15},
 'SAKT':{zh:'SAKT',g:'kt',d:'自注意力 KT：1 层、2 个头，查询是下一题的知识点，键值是做过的题。',day:16},
 'AKT':{zh:'AKT',g:'kt',d:'Rasch 风格的嵌入，加上随距离衰减的注意力。',day:17},
 'simpleKT':{zh:'simpleKT',g:'kt',d:'Rasch 风格的嵌入，加一层普通的点积注意力。',day:17},
 'DF-KT':{zh:'DF-KT',g:'df',d:'早期的决策导向 KT：BCE 加上 DR 估计的出题价值（收益是 r_gain），带策略头，没有安全屏障和方差惩罚。',day:20},
 'Safe-DFKT-AG':{zh:'Safe-DFKT-AG（本文）',g:'df',d:'本文的主方法：在 Safe-DFKT 上加反刷分惩罚（出给“预测已掌握 > 0.85”的知识点的概率）和策略头 L2。',day:20},
 'Safe-DFKT':{zh:'Safe-DFKT（去反刷分）',g:'df',abl:1,d:'消融：去掉反刷分。损失 = BCE − λ·DR(NLG) + 安全屏障 + 方差惩罚，从两阶段 DKT 热启动。',day:20},
 'Safe-DFKT w/o policy head (two-stage deploy)':{zh:'去策略头（两阶段部署）',g:'df',abl:1,d:'消融：同一个微调后的主干，扔掉策略头，只用“最接近 0.7”规则出题。用来区分“主干变了”和“策略头”各自的影响。',day:20},
 'Safe-DFKT w/o safety barrier':{zh:'去安全屏障',g:'df',abl:1,d:'消融：γ = 0，不限制 BCE 变差。',day:20},
 'Safe-DFKT w/o NLG reward':{zh:'去 NLG 收益',g:'df',abl:1,d:'消融：收益换回旧的 r_gain（下一题对错减这一题对错）。',day:19},
 'Safe-DFKT w/o POEM':{zh:'去 POEM',g:'df',abl:1,d:'消融：β = 0，不惩罚价值估计的方差。',day:19},
};
const GROUPS=[['kt','预测型 KT（两阶段：先预测，再选最接近 0.7 的）'],['ref','规则和参照'],['df','决策导向（本文和消融）']];
const GCOL={kt:'--s1',ref:'--s3',df:'--s2'};
const T1=Object.fromEntries(D.table1.map(r=>[r.method,r]));
const RES=Object.fromEntries(Object.entries(D.res).map(([d,rows])=>[d,Object.fromEntries(rows.map(r=>[r.method,r]))]));
let SEL=AG;
const f=(v,n=3)=>v==null?'—':(+v).toFixed(n),pct=(v,n=2)=>v==null?'—':(100*v).toFixed(n)+'%';
// ---------- 提示框 ----------
const tip=$('#tip');
function showTip(e,html){tip.innerHTML=html;tip.style.opacity=1;const x=Math.min(e.clientX+14,innerWidth-tip.offsetWidth-8),y=Math.min(e.clientY+14,innerHeight-tip.offsetHeight-8);tip.style.left=x+'px';tip.style.top=y+'px'}
function hideTip(){tip.style.opacity=0}
function bindTips(root){root.querySelectorAll('[data-tip]').forEach(el=>{el.addEventListener('mousemove',e=>showTip(e,el.dataset.tip));el.addEventListener('mouseleave',hideTip);el.addEventListener('focus',()=>{const r=el.getBoundingClientRect();showTip({clientX:r.right,clientY:r.top},el.dataset.tip)});el.addEventListener('blur',hideTip)})}
// ---------- 图表：横向条形图 ----------
function hbar(host,rows,o){const W=Math.max(320,host.clientWidth||600),lw=Math.min(230,W*0.42),rh=26,H=rows.length*rh+34,x0=lw+8,x1=W-56;
  const vals=rows.flatMap(r=>[r.v,r.v-(r.e||0),r.v+(r.e||0)]).filter(Number.isFinite);let lo=Math.min(0,...vals),hi=Math.max(0,...vals);if(o.lo!=null)lo=Math.min(lo,o.lo);const pad=(hi-lo)*0.05||0.01;lo-=lo<0?pad:0;hi+=pad;
  const sx=v=>x0+(v-lo)/(hi-lo)*(x1-x0),ticks=niceTicks(lo,hi,5);let tl=ticks.map(o.fmt);if(new Set(tl).size<tl.length)tl=ticks.map(t=>String(+t.toPrecision(2)));let s=`<svg width="${W}" height="${H}" role="img" aria-label="${esc(o.title||'条形图')}">`;
  ticks.forEach((t,i)=>{s+=`<line x1="${sx(t)}" x2="${sx(t)}" y1="4" y2="${H-24}" stroke="var(--line)"/><text x="${sx(t)}" y="${H-8}" text-anchor="middle">${tl[i]}</text>`});
  s+=`<line x1="${sx(0)}" x2="${sx(0)}" y1="4" y2="${H-24}" stroke="var(--muted)"/>`;
  (o.refs||[]).forEach(r=>{s+=`<line x1="${sx(r.v)}" x2="${sx(r.v)}" y1="4" y2="${H-24}" stroke="var(--ink)" stroke-dasharray="4 3"/><text x="${sx(r.v)-4}" y="${H-28}" text-anchor="end" class="lab" style="font-size:11px">${esc(r.label)}</text>`});
  rows.forEach((r,i)=>{const y=6+i*rh,col=r.col||'var(--muted)';const a=sx(Math.min(0,r.v)),b=sx(Math.max(0,r.v));
    s+=`<g tabindex="0" data-tip="${esc(r.tip)}"><rect x="0" y="${y-2}" width="${W}" height="${rh}" fill="transparent"/><text x="${lw}" y="${y+14}" text-anchor="end" class="lab"${r.bold?' font-weight="700"':''}>${esc(fitLabel(r.label,lw))}</text>`;
    if(Number.isFinite(r.v)){s+=`<rect x="${a}" y="${y+3}" width="${Math.max(1,b-a)}" height="${rh-10}" rx="3" fill="${col}"/>`;if(r.e){s+=`<line x1="${sx(r.v-r.e)}" x2="${sx(r.v+r.e)}" y1="${y+rh/2-1}" y2="${y+rh/2-1}" stroke="var(--ink)"/>`}
      const vt=o.fmt(r.v),tw=vt.length*7+4,lx=Math.min(a,r.e?sx(r.v-r.e):a)-4;s+=r.v>=0?`<text x="${Math.max(b,r.e?sx(r.v+r.e):b)+4}" y="${y+14}">${vt}</text>`:lx-tw>x0?`<text x="${lx}" y="${y+14}" text-anchor="end">${vt}</text>`:`<text x="${sx(0)+4}" y="${y+14}">${vt}</text>`}else s+=`<text x="${x0+4}" y="${y+14}">无（${esc(r.na||'不适用')}）</text>`;s+='</g>'});
  host.innerHTML=s+'</svg>';bindTips(host)}
// 标签太长时截断（完整内容在提示框里）
function fitLabel(t,w){let n=0,out='';for(const ch of t){const cw=/[\u0000-\u00ff]/.test(ch)?7:12.5;if(n+cw>w-6)return out+'…';n+=cw;out+=ch}return out}
function niceTicks(lo,hi,n){const sp=hi-lo,st0=sp/n,mg=10**Math.floor(Math.log10(st0)),e=st0/mg,st=(e>=7.5?10:e>=3.5?5:e>=1.5?2:1)*mg,t=[];for(let v=Math.ceil(lo/st)*st;v<=hi+st*1e-9;v+=st)t.push(+v.toFixed(10));return t}
// ---------- 图表：散点图 ----------
function scatter(host,pts,o){const W=Math.max(320,host.clientWidth||600),H=Math.round(Math.min(420,W*0.7)),pl=54,pr=16,pt=12,pb=40;
  const xs=pts.map(p=>p.x).filter(Number.isFinite),ys=[...pts.map(p=>p.y),...(o.hl||[]).map(h=>h.v)].filter(Number.isFinite);let[x0,x1]=[Math.min(...xs),Math.max(...xs)],[y0,y1]=[Math.min(...ys),Math.max(...ys)];const px=(x1-x0)*0.08||0.01,py=(y1-y0)*0.1||0.01;x0-=px;x1+=px;y0-=py;y1+=py;
  const sx=v=>pl+(v-x0)/(x1-x0)*(W-pl-pr),sy=v=>pt+(y1-v)/(y1-y0)*(H-pt-pb);let s=`<svg width="${W}" height="${H}" role="img" aria-label="${esc(o.title)}">`;
  niceTicks(y0,y1,5).forEach(t=>{s+=`<line x1="${pl}" x2="${W-pr}" y1="${sy(t)}" y2="${sy(t)}" stroke="var(--line)"/><text x="${pl-6}" y="${sy(t)+4}" text-anchor="end">${o.fy(t)}</text>`});
  niceTicks(x0,x1,5).forEach(t=>{s+=`<text x="${sx(t)}" y="${H-pb+16}" text-anchor="middle">${o.fx(t)}</text>`});
  s+=`<text x="${(pl+W-pr)/2}" y="${H-4}" text-anchor="middle" class="lab">${esc(o.xl)}</text><text transform="translate(12 ${(pt+H-pb)/2}) rotate(-90)" text-anchor="middle" class="lab">${esc(o.yl)}</text>`;
  (o.hl||[]).forEach(h=>{s+=`<line x1="${pl}" x2="${W-pr}" y1="${sy(h.v)}" y2="${sy(h.v)}" stroke="var(--ink)" stroke-dasharray="4 3"/><text x="${W-pr-4}" y="${sy(h.v)-5}" text-anchor="end" class="lab" style="font-size:11px">${esc(h.label)}</text>`});
  if(o.diag){const a=Math.max(x0,y0),b=Math.min(x1,y1);if(b>a)s+=`<line x1="${sx(a)}" y1="${sy(a)}" x2="${sx(b)}" y2="${sy(b)}" stroke="var(--ink)" stroke-dasharray="4 3"/><text x="${sx(b)-4}" y="${sy(b)+14}" text-anchor="end" class="lab" style="font-size:11px">y = x</text>`}
  const placed=[],tw=t=>[...t].reduce((n,ch)=>n+(/[\u0000-\u00ff]/.test(ch)?6.6:11.5),0);
  pts.filter(p=>Number.isFinite(p.x)&&Number.isFinite(p.y)).sort((a,b)=>(b.sel?1:0)-(a.sel?1:0)).forEach(p=>{const big=p.sel,cx=sx(p.x),cy=sy(p.y);let lab='';
    if(p.label){const w=tw(p.label),right=cx+9+w<=W-pr;let x=right?cx+9:cx-9,y=cy+4;const box=y0=>[right?x:x-w,y0-11,right?x+w:x,y0+3],hit=b=>placed.some(q=>b[0]<q[2]&&b[2]>q[0]&&b[1]<q[3]&&b[3]>q[1]);
      for(const dy of[0,14,-14,28,-28]){if(!hit(box(y+dy))){y+=dy;break}}placed.push(box(y));lab=`<text x="${x}" y="${y}"${right?'':' text-anchor="end"'} class="lab" style="font-size:11.5px"${big?' font-weight="700"':''}>${esc(p.label)}</text>`}
    s+=`<g tabindex="0" data-tip="${esc(p.tip)}"><circle cx="${cx}" cy="${cy}" r="14" fill="transparent"/><circle cx="${cx}" cy="${cy}" r="${big?7:5}" fill="var(${p.col})" stroke="var(--paper)" stroke-width="2"/>${lab}</g>`});
  host.innerHTML=s+'</svg>';bindTips(host)}
function lineChart(host,series,o){series=series.map(se=>({...se,pts:se.pts.filter(p=>p[1]!=null&&Number.isFinite(p[1]))})).filter(se=>se.pts.length);if(!series.length){host.innerHTML='<p class="small muted">这一项没有记录。</p>';return}const W=Math.max(320,host.clientWidth||600),H=Math.round(Math.min(o.h||340,W*0.6)),pl=54,pr=16,pt=12,pb=40;const xs=series.flatMap(s=>s.pts.map(p=>p[0])),ys=[...series.flatMap(s=>s.pts.map(p=>p[1])),...(o.hl||[]).map(h=>h.v)];
  let[x0,x1]=[Math.min(...xs),Math.max(...xs)],[y0,y1]=[Math.min(...ys),Math.max(...ys)];const py=(y1-y0)*0.1||0.01;y0-=py;y1+=py;const sx=v=>pl+(v-x0)/(x1-x0||1)*(W-pl-pr),sy=v=>pt+(y1-v)/(y1-y0)*(H-pt-pb);
  let s=`<svg width="${W}" height="${H}" role="img" aria-label="${esc(o.title)}">`;niceTicks(y0,y1,5).forEach(t=>{s+=`<line x1="${pl}" x2="${W-pr}" y1="${sy(t)}" y2="${sy(t)}" stroke="var(--line)"/><text x="${pl-6}" y="${sy(t)+4}" text-anchor="end">${o.fy(t)}</text>`});
  (o.xt||niceTicks(x0,x1,5)).forEach(t=>{s+=`<text x="${sx(t)}" y="${H-pb+16}" text-anchor="middle">${o.fx(t)}</text>`});s+=`<text x="${(pl+W-pr)/2}" y="${H-4}" text-anchor="middle" class="lab">${esc(o.xl)}</text>`;
  (o.hl||[]).forEach(h=>{s+=`<line x1="${pl}" x2="${W-pr}" y1="${sy(h.v)}" y2="${sy(h.v)}" stroke="var(--ink)" stroke-dasharray="4 3"/><text x="${W-pr-4}" y="${sy(h.v)-5}" text-anchor="end" class="lab" style="font-size:11px">${esc(h.label)}</text>`});
  (o.vl||[]).forEach((v,i)=>{const X=sx(v.x),R=X>W*0.7;s+=`<line x1="${X}" x2="${X}" y1="${pt}" y2="${H-pb}" stroke="var(${v.col||'--ink'})" stroke-width="1.5" stroke-dasharray="${v.dash||'4 3'}"/>${v.label?`<text x="${X+(R?-4:4)}" y="${pt+12+i*13}"${R?' text-anchor="end"':''} class="lab" style="font-size:11px">${esc(v.label)}</text>`:''}`});
  series.forEach(se=>{s+=`<polyline fill="none" stroke="var(${se.col})" stroke-width="2" points="${se.pts.map(p=>sx(p[0])+','+sy(p[1])).join(' ')}"/>`;se.pts.forEach(p=>{s+=`<g tabindex="0" data-tip="${esc(se.name+'：'+o.fx(p[0])+' → '+o.fy(p[1]))}"><circle cx="${sx(p[0])}" cy="${sy(p[1])}" r="12" fill="transparent"/><circle cx="${sx(p[0])}" cy="${sy(p[1])}" r="4" fill="var(${se.col})" stroke="var(--paper)" stroke-width="2"/></g>`})});
  host.innerHTML=s+'</svg>';bindTips(host)}
const legend=(host,items)=>{host.innerHTML=items.map(([c,t])=>`<span><i style="background:var(${c})"></i>${esc(t)}</span>`).join('')};
// ---------- 一页看懂 ----------
function heroTiles(){const d=T1.DKT,a=T1[AG],s=T1['Safe-DFKT'],r=T1.Random;
  const t=[['预测 AUC',`${pct(a.auc)} <small>DKT ${pct(d.auc)}</small>`,'预测几乎没变差','tag good','保住了'],['校准误差 ECE',`${pct(a.ece)} <small>DKT ${pct(d.ece)}</small>`,'越小越好','tag good','保住了'],['模拟学习收益 NLG',`${f(a.sim_nlg,4)} <small>DKT ${f(d.sim_nlg,4)}</small>`,'越大越好','tag bad','没有提高'],['出给已掌握知识点',`${f(a.sim_pct_sel_mastered,1)}% <small>去反刷分 ${f(s.sim_pct_sel_mastered,1)}%</small>`,`DKT ${f(d.sim_pct_sel_mastered,1)}%`,'tag bad','刷分'],['随机出题的 NLG',`${f(r.sim_nlg,4)}`,'所有方法里最高之一','tag','意外']];
  $('#heroTiles').innerHTML=t.map(([k,v,n,c,b])=>`<div class="tile"><span class="k">${k} <span class="${c}">${b}</span></span><span class="v">${v}</span><span class="src">${n} · 4 个数据集 × 5 个种子的平均</span></div>`).join('')}
// ---------- 数据集 ----------
let DS='assist2009';
function dsRender(){const d=D.datasets.find(x=>x.dataset===DS);$$('#dsChips .chip').forEach(c=>c.setAttribute('aria-pressed',c.dataset.ds===DS));$('#dsTitle').textContent=DSN[DS];
  const w=d.real_windows,tiles=[['学生',d.n_students.toLocaleString()],['做题记录',d.n_tokens.toLocaleString()],['知识点 K',d.K],['训练 / 验证 / 测试窗口',`${w.train} / ${w.val} / ${w.test}`],['先修关系',d.n_prereq_edges_total||'无'],['模拟器：随机出题 NLG',f(d.ref_random.nlg,3)]];
  $('#dsTiles').innerHTML=tiles.map(([k,v])=>`<div class="tile"><span class="k">${k}</span><span class="v" style="font-size:19px">${v}</span></div>`).join('');
  $('#dsNote').textContent=`每个学生最多保留最近 500 步，按 100 步切成窗口。模拟器从这份日志里挑了 12 个知识点${DS==='junyi'?'（沿先修关系挑一组相连的，模拟器里有 '+d.sim.n_prereq_edges_in_sim+' 条先修关系）':'（按出现次数）'}；旧日志策略在模拟器里的 NLG 是 ${f(d.ref_logging.nlg,3)}。`;parRender()}
function parRender(){const d=D.datasets.find(x=>x.dataset===DS).sim,k=$('#parSel').value,v=d[k];hbar($('#parChart'),v.map((x,i)=>({label:`知识点 ${i+1}（#${d.skills[i]}）`,v:x,col:'var(--s1)',tip:`知识点 ${i+1}（原编号 ${d.skills[i]}）<br>p0 ${d.p0[i]} · learn ${d.learn[i]} · guess ${d.guess[i]} · slip ${d.slip[i]}`})),{fmt:x=>x.toFixed(2),title:'模拟器参数'})}
// ---------- 模型 ----------
function modelsRender(){$('#modelGroups').innerHTML=GROUPS.map(([g,t])=>`<div class="stack"><h3>${t}</h3><div class="models">${Object.keys(M).filter(m=>M[m].g===g).map(m=>{const r=T1[m]||{};return`<button class="mcard" data-m="${esc(m)}" aria-pressed="${m===SEL}"><span class="row" style="gap:6px"><span class="nm">${esc(M[m].zh)}</span>${m===AG?'<span class="tag ours">本文</span>':''}${M[m].abl?'<span class="tag">消融</span>':''}</span><span class="small">${esc(M[m].d)}</span><span class="nums">AUC ${r.auc!=null?pct(r.auc):'—'} · NLG ${f(r.sim_nlg,3)}${r.params?'':''}</span>${M[m].day?`<a class="small" href="${COURSE}${M[m].day}" target="_blank" rel="noopener">课程第 ${M[m].day} 天</a>`:''}</button>`}).join('')}</div></div>`).join('');
  $$('.mcard').forEach(b=>b.addEventListener('click',e=>{if(e.target.tagName==='A')return;SEL=b.dataset.m;$$('.mcard').forEach(x=>x.setAttribute('aria-pressed',x.dataset.m===SEL));allCharts()}))}
// ---------- 结果浏览器 ----------
const METS=[['sim_nlg','模拟学习收益 NLG',1,v=>v.toFixed(3),'前测到后测提高了多少，按“还能提高的空间”归一化。越大越好。'],['auc','预测 AUC（真实日志）',1,v=>(100*v).toFixed(1)+'%','随机挑一道答对和一道答错的题，模型给答对那道更高分的概率。越大越好。'],['ece','校准误差 ECE（真实日志）',-1,v=>(100*v).toFixed(2)+'%','预测概率和实际答对率的平均差距。越小越好。'],['sim_pct_sel_mastered','出给已掌握知识点的比例',-1,v=>v.toFixed(1)+'%','模拟器里每一步出的题，学生其实已经掌握的比例。越小说明越多地在教新东西。'],['sim_stm','掌握所需步数',-1,v=>v.toFixed(1),'连续答对 3 次算掌握，平均要多少步（没掌握按 61 计）。越小越好。'],['sim_mastery_gain','掌握的知识点增加比例',1,v=>v.toFixed(3),'练习后掌握的知识点比例减去练习前。越大越好。'],['dec_sim_nlg','NLG（按决策价值选的检查点）',1,v=>v.toFixed(3),'只有决策导向方法有：按模拟验证队列的真实 NLG 选检查点时的结果。'],['sim_auc','模拟器里的预测 AUC',1,v=>(100*v).toFixed(1)+'%','在模拟日志上训练、模拟学生上测的 AUC。'],['ope_dr','真实日志上的 OPE（DR）',1,v=>v.toFixed(3),'用离线策略评估在真实日志上估计的出题价值。'],['sim_ope_bias','OPE 偏差（模拟器）',0,v=>v.toFixed(3),'模拟器里 DR 估计减去真实值。越接近 0 越好。']];
function rRender(){const ds=$('#rDs').value,mk=$('#rMet').value,meta=METS.find(m=>m[0]===mk),showAbl=$('#rAbl').value==='1';const src=ds==='all'?T1:RES[ds];
  let ms=Object.keys(M).filter(m=>showAbl||!M[m].abl);const sd=r=>r?(r[mk+'_std']!=null?r[mk+'_std']:null):null;
  let rows=ms.map(m=>{const r=src[m],v=r?r[mk]:null,e=sd(r);return{m,label:M[m].zh,v:v==null?NaN:v,e:e||0,bold:m===SEL,col:m===SEL?'var(--s2)':m==='DKT'?'var(--s1)':'color-mix(in srgb,var(--muted) 55%,var(--paper))',na:m==='Random'?'随机出题没有预测':'只有决策导向方法有',tip:`<b>${esc(M[m].zh)}</b><br>${esc(meta[1])}：${Number.isFinite(v)?meta[3](v):'无'}${e?' ± '+meta[3](e):''}<br>${ds==='all'?'4 个数据集 × 5 个种子':'5 个种子'}`}});
  const dir=meta[2];rows.sort((a,b)=>(Number.isFinite(b.v)-Number.isFinite(a.v))||(dir===0?Math.abs(a.v)-Math.abs(b.v):dir*(b.v-a.v)));
  const refs=mk==='sim_nlg'&&src.Random?[{v:src.Random.sim_nlg,label:'随机出题'}]:[];
  $('#rMetNote').innerHTML=`<b>${meta[1]}</b>：${meta[4]} 已按${dir===0?'离 0 的远近':'好坏'}排序，最好的在上面。`;hbar($('#rChart'),rows,{fmt:meta[3],refs,title:meta[1]})}
// ---------- 预测好≠出题好 ----------
function tRender(){const ds=$('#tDs').value,src=ds==='all'?T1:RES[ds];const pts=Object.keys(M).filter(m=>!M[m].abl&&m!=='Random').map(m=>{const r=src[m];return{x:r?r.auc:NaN,y:r?r.sim_nlg:NaN,col:GCOL[M[m].g],sel:m===SEL,label:['DKT',AG,'Uncertainty-0.5','Heuristic-0.7','LR','SAKT'].includes(m)||m===SEL?M[m].zh:'',tip:`<b>${esc(M[m].zh)}</b><br>AUC ${r?pct(r.auc,1):'—'} · NLG ${r?f(r.sim_nlg,3):'—'}`}});
  scatter($('#tChart'),pts,{xl:'预测 AUC（真实日志）',yl:'模拟学习收益 NLG',fx:v=>(100*v).toFixed(1)+'%',fy:v=>v.toFixed(2),hl:src.Random?[{v:src.Random.sim_nlg,label:'随机出题（没有 AUC）'}]:[],title:'AUC 和 NLG'});
  const dk=src.DKT,un=src['Uncertainty-0.5'];$('#tNote').innerHTML=`同一个 DKT 的预测，只把目标从 0.7 改成 0.5（Uncertainty-0.5），AUC 完全一样（${pct(dk.auc,1)}），NLG 从 ${f(dk.sim_nlg,3)} 变成 ${f(un.sim_nlg,3)}。决定出题效果的主要是<b>选题规则</b>，而不是预测准不准。`}
// ---------- 刷分诊断 ----------
function hRender(){const ds=$('#hDs').value,src=ds==='all'?T1:RES[ds];const pts=Object.keys(M).filter(m=>src[m]&&src[m].sim_pct_sel_mastered!=null).map(m=>{const r=src[m];return{x:r.sim_pct_sel_mastered,y:r.sim_nlg,col:GCOL[M[m].g],sel:m===SEL,label:['DKT',AG,'Safe-DFKT','Random','Uncertainty-0.5'].includes(m)||m===SEL?M[m].zh:'',tip:`<b>${esc(M[m].zh)}</b><br>已掌握 ${f(r.sim_pct_sel_mastered,1)}% · NLG ${f(r.sim_nlg,3)}`}});
  scatter($('#hChart'),pts,{xl:'出给已掌握知识点的比例',yl:'NLG',fx:v=>v.toFixed(0)+'%',fy:v=>v.toFixed(2),title:'刷分'});legend($('#hLegend'),GROUPS.map(([g,t])=>[GCOL[g],t]));
  const rows=D.decomp.filter(r=>r.sim_flip_backbone!=null).map(r=>[r,M[r.method]?M[r.method].zh:r.method]);
  hbar($('#dcChart'),rows.flatMap(([r,n])=>[{label:n+'：主干',v:r.sim_flip_backbone,col:'var(--s1)',tip:`${esc(n)}<br>主干改变选择 ${pct(r.sim_flip_backbone,1)}`},...(r.sim_flip_head!=null?[{label:n+'：策略头',v:r.sim_flip_head,col:'var(--s2)',tip:`${esc(n)}<br>策略头改变选择 ${pct(r.sim_flip_head,1)}`}]:[])]),{fmt:v=>(100*v).toFixed(0)+'%',title:'改变选择的比例'});
  const c=D.calgap[0];$('#calNote').innerHTML=`<b>重新校准能补救吗？</b>把 Safe-DFKT 的预测用 Platt 缩放重新校准后，它和 DKT 的 NLG 差距从 ${f(c.nlg_gap_dkt_minus_safe_raw,3)} 只缩小到 ${f(c.nlg_gap_dkt_minus_safe_recal,3)}，只补回了 ${(100*c.frac_nlg_gap_closed_by_recalibration).toFixed(1)}%。问题不在校准，而在策略头学到的选择。`}
// ---------- 稳健性 ----------
function rbRender(){const ms=['DKT',AG,'Safe-DFKT'],cols=['--s1','--s2','--s3'];lineChart($('#rbTarget'),ms.map((m,i)=>({name:M[m].zh,col:cols[i],pts:D.target.filter(r=>r.method===m).map(r=>[r.c_target,r.sim_nlg]).sort((a,b)=>a[0]-b[0])})),{xl:'目标答对率 c',fx:v=>v.toFixed(1),fy:v=>v.toFixed(2),xt:[0.3,0.5,0.7],hl:[{v:D.target[0].random_sim_nlg,label:'随机出题'}],title:'c 扫描'});legend($('#rbTargetLeg'),ms.map((m,i)=>[cols[i],M[m].zh]));
  $('#rbCkpt').innerHTML=D.ckpt.map(r=>`<tr class="${r.method===AG?'hl':''}"><td>${esc(M[r.method]?M[r.method].zh:r.method)}</td><td class="mono">${r.criterion}</td><td class="n">${f(r.sim_nlg,3)}</td><td class="n">${pct(r.auc,1)}</td></tr>`).join('');
  $('#rbEps').innerHTML=D.eps.map(r=>`<tr><td class="n">${r.epsilon}</td><td class="n">${f(r.sim_nlg,3)}</td><td class="n">${pct(r.auc,1)}</td><td class="n">${pct(r.ece,2)}</td><td class="n">${f(r.sim_pct_sel_mastered,1)}%</td></tr>`).join('');
  const sm=['Random','DKT',AG],sc=['--s3','--s1','--s2'];lineChart($('#rbScale'),sm.map((m,i)=>({name:M[m].zh,col:sc[i],pts:D.scale.filter(r=>r.method===m).map(r=>[r.n_skills,r.sim_nlg])})),{xl:'模拟器的知识点个数',fx:v=>String(v),fy:v=>v.toFixed(2),xt:[12,24],title:'知识点规模'});legend($('#rbScaleLeg'),sm.map((m,i)=>[sc[i],M[m].zh]))}
// ---------- 统计检验 ----------
const MZ={sim_nlg:'NLG',auc:'AUC',ece:'ECE',sim_ece:'模拟器 ECE',sim_pct_sel_mastered:'已掌握%',sim_stm:'掌握步数'};
function stRender(){const c=$('#stCmp').value,rows=D.paired.filter(r=>r.comparison===c);const fm=(m,v)=>m==='auc'||m==='ece'||m==='sim_ece'?(100*v).toFixed(2)+'%':m==='sim_pct_sel_mastered'?v.toFixed(1)+'%':m==='sim_stm'?v.toFixed(2):v.toFixed(4);
  $('#stTbl').innerHTML=rows.map(r=>{const p=r.p_wilcoxon_holm;return`<tr><td>${MZ[r.metric]||r.metric}</td><td class="n">${fm(r.metric,r.mean_a)}</td><td class="n">${fm(r.metric,r.mean_b)}</td><td class="n">${fm(r.metric,r.mean_diff)}</td><td class="n">[${fm(r.metric,r.ci95_low)}, ${fm(r.metric,r.ci95_high)}]</td><td class="n">${(100*r.frac_a_better).toFixed(0)}%</td><td class="n">${p==null?'—':p<1e-4?'&lt; 10⁻⁴':p.toFixed(4)}${p!=null&&p<0.05?' <span class="tag">显著</span>':''}</td></tr>`}).join('')}
// ---------- 结论 ----------
function conclRender(){const a=T1[AG],d=T1.DKT,c=D.calgap[0];$('#concl').innerHTML=[`预测能力保住了：AUC ${pct(a.auc)} 对 DKT ${pct(d.auc)}，ECE ${pct(a.ece)} 对 ${pct(d.ece)}。`,`出题效果没有提高：NLG ${f(a.sim_nlg,4)} 对 DKT ${f(d.sim_nlg,4)}；按决策价值选检查点也只有 ${f(a.dec_sim_nlg,4)}。`,`原因是刷分：去掉反刷分时 ${f(T1['Safe-DFKT'].sim_pct_sel_mastered,1)}% 的题出给了已掌握的知识点（DKT ${f(d.sim_pct_sel_mastered,1)}%）；反刷分惩罚降到 ${f(a.sim_pct_sel_mastered,1)}%，但仍落后 DKT。`,`重新校准只补回 ${(100*c.frac_nlg_gap_closed_by_recalibration).toFixed(1)}% 的差距，差距主要来自策略头。`,`在这个环境里，随机出题（${f(T1.Random.sim_nlg,4)}）和更低的目标答对率表现最好，因为对所有知识点取平均的 NLG 奖励覆盖面。`,`一句话：保住校准本身，不足以让决策变好。`].map(t=>`<li>${t}</li>`).join('')}
// ---------- 模拟器游乐场 ----------
const HIST=[];
function polOf(){const k=$('#sPol').value,c=+$('#sC').value;if(k==='random')return[POL.random(),'随机'];if(k==='logging')return[POL.logging(),'旧日志策略'];if(k==='greedy')return[POL.target(0.999,'emp'),'总出最有把握的'];if(k==='target-oracle')return[POL.target(c,'oracle'),`理想模型，c = ${c.toFixed(2)}`];return[POL.target(c,'emp'),`经验正确率，c = ${c.toFixed(2)}`]}
function simRun(){const ds=$('#sDs').value,sim=makeSim(D.datasets.find(x=>x.dataset===ds).sim),n=+$('#sN').value,[pol,name]=polOf();const r=rollout(sim,pol,n,60,HIST.length+11);
  $('#sTiles').innerHTML=[['NLG',f(r.nlg,3)],['出给已掌握的',(100*r.fracMastered).toFixed(1)+'%'],['掌握所需步数',r.stm.toFixed(1)],['每人练到的知识点',r.distinct.toFixed(1)+' / 12']].map(([k,v])=>`<div class="tile"><span class="k">${k}</span><span class="v">${v}</span></div>`).join('');
  const t=r.traj;$('#sTraj').innerHTML=t.map(([a,y,Lk,lr],i)=>`<div class="${y?'ok':'no'} ${Lk?'m':''} ${lr?'lr':''}" title="第 ${i+1} 题：知识点 ${a+1}，${y?'对':'错'}${Lk?'，已掌握':''}${lr?'，这一题学会了':''}">${a+1}</div>`).join('');
  HIST.unshift({ds,name,nlg:r.nlg,m:r.fracMastered,d:r.distinct});if(HIST.length>8)HIST.pop();histRender();
  $('#sNote').textContent=r.fracMastered>0.85?'大部分题出给了已经会的知识点：做对率很高，但几乎没学新东西，NLG 很低甚至为负（遗忘让后测下降）。这就是论文说的刷分。':r.distinct<5?'只在少数几个知识点上反复练，其他知识点没碰，而 NLG 是 12 个知识点的平均。':'题目分散到了多数知识点上，没掌握的也练到了，NLG 比较高。'}
function histRender(){$('#sTbl').innerHTML=HIST.map(h=>`<tr><td>${DSN[h.ds]}</td><td>${esc(h.name)}</td><td class="n">${f(h.nlg,3)}</td><td class="n">${(100*h.m).toFixed(1)}%</td><td class="n">${h.d.toFixed(1)}</td></tr>`).join('');
  hbar($('#sHist'),HIST.map((h,i)=>({label:`${i+1}. ${h.name}`,v:h.nlg,col:i===0?'var(--s2)':'var(--s1)',tip:`${DSN[h.ds]} · ${esc(h.name)}<br>NLG ${f(h.nlg,3)}`})),{fmt:v=>v.toFixed(2),title:'运行记录'})}
function simSweep(){const ds=$('#sDs').value,sim=makeSim(D.datasets.find(x=>x.dataset===ds).sim),cs=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95];const se=[['emp','经验正确率','--s1'],['oracle','理想模型','--s2']].map(([k,n,col])=>({name:n,col,pts:cs.map(c=>[c,rollout(sim,POL.target(c,k),300,60,7).nlg])}));
  const rnd=rollout(sim,POL.random(),300,60,7).nlg;$('#sweepBox').hidden=false;lineChart($('#sSweepChart'),se,{xl:'目标答对率 c',fx:v=>v.toFixed(1),fy:v=>v.toFixed(2),hl:[{v:rnd,label:'随机出题'}],title:'c 扫描'});legend($('#sSweepLeg'),se.map(s=>[s.col,s.name]))}
function simValid(){const rows=D.datasets.map(d=>{const sim=makeSim(d.sim);return`${DSN[d.dataset]}：随机 ${f(rollout(sim,POL.random(),600,60,1).nlg,3)}（论文 ${f(d.ref_random.nlg,3)}）`});
  $('#simValid').textContent=`和论文对照（随机出题的 NLG，浏览器 600 个学生 vs 论文的参照值）：${rows.join('；')}。ASSISTments、EdNet、Algebra 基本一致；Junyi 的模拟器有先修关系，但结果文件里没有保存具体是哪几条，这里没法复现，所以 Junyi 的数字会偏高。另外“经验正确率”的先验用的是模拟器参数算出的平均答对率，论文用的是模拟日志的统计，略有不同。`}
// ---------- 离线评估 ----------
function oRender(){const ds=$('#oDs').value,src=RES[ds];const ms=Object.keys(M).filter(m=>src[m]&&src[m].ope_dr!=null);
  $('#oTbl').innerHTML=ms.map(m=>{const r=src[m];return`<tr class="${m===SEL?'hl':''}"><td>${esc(M[m].zh)}</td><td class="n">${f(r.ope_ips,3)}</td><td class="n">${f(r.ope_snips,3)}</td><td class="n">${f(r.ope_dr,3)} ± ${f(r.ope_dr_se,3)}</td><td class="n">${f(r.ope_ess,3)}</td></tr>`}).join('');
  const ess=ms.map(m=>src[m].ope_ess).filter(v=>v!=null);$('#oNote').innerHTML=`ESS（归一化的有效样本量）在 ${f(Math.min(...ess),3)} 到 ${f(Math.max(...ess),3)} 之间：新规则和收集日志的旧规则差得很远，几千条记录里真正起作用的只有一小部分。所以 IPS（不归一化）和 SNIPS、DR 差得很大。真实日志上的 OPE 只能当辅助证据，论文的主结论用的是模拟器里的真值。`;
  const pts=ms.filter(m=>src[m].sim_ope_dr!=null&&src[m].sim_ope_true!=null).map(m=>{const r=src[m];return{x:r.sim_ope_dr,y:r.sim_ope_true,col:GCOL[M[m].g],sel:m===SEL,label:['DKT',AG,'Random'].includes(m)||m===SEL?M[m].zh:'',tip:`<b>${esc(M[m].zh)}</b><br>DR 估计 ${f(r.sim_ope_dr,3)}<br>真实值 ${f(r.sim_ope_true,3)}<br>偏差 ${f(r.sim_ope_bias,3)} · ESS ${f(r.sim_ess,3)}`}});
  scatter($('#oChart'),pts,{xl:'DR 估计',yl:'真实值',fx:v=>v.toFixed(2),fy:v=>v.toFixed(2),diag:true,title:'OPE 估计和真值'});legend($('#oLeg'),GROUPS.map(([g,t])=>[GCOL[g],t]))}
// ---------- 学生回放 ----------
const RP={seed:3,t:0,timer:null,A:null,B:null};
const RPOL=[['target-emp-0.7','选最接近 0.7 的（两阶段 DKT 的规则）'],['greedy','总出最有把握答对的（模拟刷分）'],['random','完全随机'],['target-emp-0.5','选最接近 0.5 的'],['target-emp-0.3','选最接近 0.3 的'],['target-oracle-0.7','理想模型，选最接近 0.7 的'],['logging','旧日志策略']];
function rpPol(k){if(k==='random')return POL.random();if(k==='logging')return POL.logging();if(k==='greedy')return POL.target(0.999,'emp');const[,est,c]=k.split('-');return POL.target(+c,est)}
// 挑一个“一开始会的知识点不超过 6 个”的学生，这样才有东西可学
function rpNextSeed(sim,from){for(let sd=from;sd<from+200;sd++){if(rolloutOne(sim,POL.random(),sd,0).steps[0].L.filter(Boolean).length<=6)return sd}return from}
function rpRun(adv){const ds=$('#rpDs').value,sim=makeSim(D.datasets.find(x=>x.dataset===ds).sim);RP.sim=sim;RP.seed=rpNextSeed(sim,adv?RP.seed+1:RP.seed);RP.A=rolloutOne(sim,rpPol($('#rpA').value),RP.seed);RP.B=rolloutOne(sim,rpPol($('#rpB').value),RP.seed);rpDraw()}
function rpPanel(host,run,name){const t=RP.t,st=run.steps[t],pre=run.steps[0].perf,K=RP.sim.K,cnt=new Array(K).fill(0);let onM=0,learned=0,forgot=0;for(let i=1;i<=t;i++){const s=run.steps[i];cnt[s.a]++;if(s.Lk)onM++;if(s.learn)learned++;forgot+=s.forgot.length}
  const cells=st.L.map((m,k)=>`<div class="sk ${m?'m':''} ${t>0&&st.a===k?'cur':''}" title="知识点 ${k+1}：${m?'已掌握':'未掌握'}，练了 ${cnt[k]} 次"><b>${k+1}</b><span>${cnt[k]} 次</span></div>`).join('');
  const nlg=(st.perf-pre)/Math.max(1-pre,1e-3),last=t>0?`第 ${t} 题出了知识点 ${st.a+1}，${st.Lk?'学生已经会了':'学生还不会'}，${st.y?'答对':'答错'}${st.learn?'，这一题学会了':''}${st.forgot.length?`；同时忘了知识点 ${st.forgot.map(k=>k+1).join('、')}`:''}。`:'还没开始做题。';
  host.innerHTML=`<h3>${esc(name)}</h3><div class="skgrid">${cells}</div><div class="tiles"><div class="tile"><span class="k">测验分（12 个知识点平均答对率）</span><span class="v" style="font-size:19px">${(100*st.perf).toFixed(1)}%<small>开始时 ${(100*pre).toFixed(1)}%</small></span></div><div class="tile"><span class="k">到目前的 NLG</span><span class="v" style="font-size:19px">${nlg.toFixed(3)}</span></div><div class="tile"><span class="k">出给已掌握的题</span><span class="v" style="font-size:19px">${onM} / ${t}</span></div><div class="tile"><span class="k">学会 / 遗忘次数</span><span class="v" style="font-size:19px">${learned} / ${forgot}</span></div></div><p class="small">${last}</p>`}
function rpDraw(){const t=RP.t;$('#rpTv').textContent=t;$('#rpT').value=t;const na=$('#rpA').selectedOptions[0].text,nb=$('#rpB').selectedOptions[0].text;rpPanel($('#rpPA'),RP.A,'左：'+na);rpPanel($('#rpPB'),RP.B,'右：'+nb);
  const a=RP.A.steps[60],b=RP.B.steps[60],p0=RP.A.steps[0].perf,nA=(a.perf-p0)/(1-p0),nB=(b.perf-p0)/(1-p0);
  $('#rpNote').textContent=`这个学生能力 θ = ${RP.A.theta.toFixed(2)}，一开始会 ${RP.A.steps[0].L.filter(Boolean).length} 个知识点。做完 60 题：左边 NLG ${nA.toFixed(3)}，右边 ${nB.toFixed(3)}。一个学生的结果偶然性很大，多换几个学生看看；模拟器游乐场里是几百个学生的平均。`}
function rpPlay(){if(RP.timer){clearInterval(RP.timer);RP.timer=null;$('#rpPlay').textContent='播放';return}if(RP.t>=60)RP.t=0;$('#rpPlay').textContent='暂停';RP.timer=setInterval(()=>{RP.t++;rpDraw();if(RP.t>=60){clearInterval(RP.timer);RP.timer=null;$('#rpPlay').textContent='播放'}},matchMedia('(prefers-reduced-motion: reduce)').matches?600:260)}
// ---------- 训练曲线 ----------
function ckRender(){const ds=$('#ckDs').value,m=$('#ckM').value,w=$('#ckW').value,sd=+$('#ckS').value,r=D.traces.find(x=>x.ds===ds&&x.m===m&&x.w===w&&x.seed===sd);if(!r){$('#ckBce').innerHTML=$('#ckDec').innerHTML='<p class="small muted">没有这一组记录。</p>';return}
  const ep=r.bce.map((_,i)=>i+1),vl=[{x:r.sb+1,label:`BCE 选第 ${r.sb+1} 轮`,col:'--s1'}];if(r.sd!=null)vl.push({x:r.sd+1,label:`决策价值选第 ${r.sd+1} 轮`,col:'--s2',dash:'2 3'});
  lineChart($('#ckBce'),[{name:'验证集 BCE',col:'--s1',pts:ep.map((e,i)=>[e,r.bce[i]])}],{xl:'训练轮数',fx:v=>String(v),fy:v=>v.toFixed(4),vl,h:260,title:'验证 BCE'});
  $('#ckDecT').textContent=w==='sim'?'验证队列的真实 NLG（越高越好）':'验证日志的 DR 估计（越高越好）';
  lineChart($('#ckDec'),[{name:w==='sim'?'真实 NLG':'DR',col:'--s2',pts:ep.map((e,i)=>[e,r.dec[i]])}],{xl:'训练轮数',fx:v=>String(v),fy:v=>v.toFixed(3),vl,h:260,title:'决策价值'});
  const hasDec=r.dec.some(v=>v!=null);$('#ckNote').textContent=!hasDec?'这个消融实验没有记录决策价值，只按验证集 BCE 选检查点。':r.sd===r.sb?'两种标准选中了同一轮。':`两种标准选中的轮数不一样（第 ${r.sb+1} 轮和第 ${r.sd+1} 轮）：预测最好的那一轮，出题不一定最好。`}
// ---------- 术语 ----------
const TERMS=[['知识追踪（KT）','根据学生过去的做题记录，预测下一题能不能答对。',14],['BKT','贝叶斯知识追踪：每个知识点只有“会”和“不会”，用 p0、learn、guess、slip 四个参数描述。模拟器就是按它造的。',4],['AUC','随机挑一道答对和一道答错的题，模型给答对那道更高分的概率。0.5 是瞎猜。',14],['BCE','二元交叉熵，预测对错时的损失。验证集 BCE 用来选检查点。',6],['ECE','期望校准误差：把预测概率分成 10 段，每段里预测概率和实际答对率差多少，加权平均。',6],['两阶段','先训练一个预测模型，再用固定规则（选最接近 0.7 的）出题。DKT、SAKT 等基线都是这样。',8],['决策导向（DF）','训练时直接把“出题价值”放进损失，让模型学会出题，而不只是预测。',20],['NLG','归一化学习收益：(后测 − 前测) / (1 − 前测)。论文对 12 个知识点取平均，所以覆盖面越广越占便宜。',20],['掌握所需步数','连续答对 3 次算掌握，平均要多少步；没掌握的按 61 计。',20],['OPE','离线策略评估：只用旧策略收集的日志，估计新策略的价值。',19],['IPS / SNIPS','逆倾向加权：用 π/μ 给每条日志加权。SNIPS 用权重之和代替 n 做分母，方差更小。',19],['DR','双重稳健估计：结果模型的预测加上加权残差。权重或结果模型有一个对就无偏。',19],['ESS','有效样本量：(Σw)²/Σw²/n。越接近 0，估计越依赖少数几条记录。',19],['交叉拟合','学生分两折，一折用的 μ 和 Q 在另一折上估计，避免“自己评自己”的偏差。',19],['一步偏离','只在一个决策点换成新策略、之后照旧。论文的 DR 估计的就是这个价值。',19],['安全屏障','γ·ReLU(BCE − (1+ε)·BCE_DKT)：预测变差超过 ε 才惩罚。',20],['POEM 方差惩罚','β·Var(V)：惩罚价值估计的方差，避免策略追着噪声大的估计跑。',20],['反刷分（AG）','惩罚把出题概率放在“预测已掌握（> 0.85）”的知识点上。',20],['刷分（reward hacking）','模型找到让收益估计变高、却不真正提高学习的捷径：总出已经会的题。',20],['Platt 缩放','对 logit 做 a·z + b 再过 sigmoid，重新校准预测概率。',6],['配对检验和 Holm 校正','同一个种子和数据集上两个方法配成一对比较差值；做多次检验时用 Holm 方法收紧 p 值门槛。',20]];
function termsRender(){$('#termList').innerHTML=TERMS.map(([t,d,day])=>`<div class="mcard" style="cursor:default"><span class="nm">${esc(t)}</span><span class="small">${esc(d)}</span><a class="small" href="${COURSE}${day}" target="_blank" rel="noopener">课程第 ${day} 天</a></div>`).join('')}
// ---------- 启动 ----------
function allCharts(){rRender();tRender();hRender();rbRender();parRender();oRender();if(RP.A)rpDraw();ckRender()}
function init(){heroTiles();
  $('#dsChips').innerHTML=DSL.map(d=>`<button class="chip" data-ds="${d}">${DSN[d]}</button>`).join('');$$('#dsChips .chip').forEach(c=>c.onclick=()=>{DS=c.dataset.ds;dsRender()});$('#parSel').onchange=parRender;
  const dsOpts=(all)=>(all?'<option value="all">四个数据集平均</option>':'')+DSL.map(d=>`<option value="${d}">${DSN[d]}</option>`).join('');
  ['#rDs','#tDs','#hDs'].forEach(s=>{$(s).innerHTML=dsOpts(true);$(s).onchange=allCharts});['#sDs','#oDs','#rpDs','#ckDs'].forEach(s=>$(s).innerHTML=dsOpts(false));$('#oDs').onchange=oRender;
  $('#rpA').innerHTML=$('#rpB').innerHTML=RPOL.map(([k,n])=>`<option value="${k}">${esc(n)}</option>`).join('');$('#rpB').value='random';['#rpDs','#rpA','#rpB'].forEach(s=>$(s).onchange=()=>{rpRun()});$('#rpNew').onclick=()=>{rpRun(true)};$('#rpPlay').onclick=rpPlay;$('#rpT').oninput=()=>{RP.t=+$('#rpT').value;rpDraw()};
  $('#ckM').innerHTML=['Safe-DFKT-AG','Safe-DFKT','DF-KT','Safe-DFKT w/o safety barrier','Safe-DFKT w/o NLG reward','Safe-DFKT w/o POEM'].map(m=>`<option value="${esc(m)}">${esc(M[m].zh)}</option>`).join('');['#ckDs','#ckM','#ckW','#ckS'].forEach(s=>$(s).onchange=ckRender);
  $('#rMet').innerHTML=METS.map(m=>`<option value="${m[0]}">${m[1]}</option>`).join('');$('#rMet').onchange=rRender;$('#rAbl').onchange=rRender;
  $('#stCmp').innerHTML=[...new Set(D.paired.map(r=>r.comparison))].map(c=>`<option>${esc(c)}</option>`).join('');$('#stCmp').onchange=stRender;
  $('#sC').oninput=()=>$('#sCv').textContent=(+$('#sC').value).toFixed(2);$('#sN').oninput=()=>$('#sNv').textContent=$('#sN').value;
  $('#sRun').onclick=simRun;$('#sSweep').onclick=simSweep;$('#sClear').onclick=()=>{HIST.length=0;histRender()};
  modelsRender();dsRender();RP.t=60;rpRun();allCharts();stRender();conclRender();termsRender();simRun();simValid();
  let rz;addEventListener('resize',()=>{clearTimeout(rz);rz=setTimeout(()=>{allCharts();histRender()},150)});
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change',allCharts)}
init();
