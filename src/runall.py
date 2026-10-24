import os as _o0,base64 as _b0
def _rawb64():
    p=_o0.path.join(_o0.path.dirname(_o0.path.abspath(__file__)),'fmnist_raw_b64.txt')
    if not _o0.path.exists(p):
        import numpy as np
        z=np.load(p.replace('fmnist_raw_b64.txt','fmnist_subset.npz'))
        open(p,'w').write(_b0.b64encode(np.concatenate([z['Xtr'].ravel(),z['ytr'],z['Xte'].ravel(),z['yte']]).astype(np.uint8).tobytes()).decode())
    return open(p).read()
import re,html,json,subprocess
import os as _os;d=_os.path.dirname(_os.path.abspath(__file__))+'/'
b=open(d+'page_body2.html').read()
days={}
for m in re.finditer(r'<main class="wrap day" id="day(\d+)".*?</main>',b,re.S):
    days[m.group(1)]=[html.unescape(t) for t in re.findall(r'<div class="cell"(?![^>]*data-manual)[^>]*><textarea>(.*?)</textarea>',m.group(0),re.S)]
js=open(d+'minitorch2.js').read()
pre="var B64='"+_rawb64()+"';var _A='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';var _L=new Uint8Array(256);for(var i=0;i<64;i++)_L[_A.charCodeAt(i)]=i;var _n=B64.length*3/4-(B64.endsWith('==')?2:B64.endsWith('=')?1:0);var U=new Uint8Array(_n);var p=0;for(var i=0;i<B64.length;i+=4){var a=_L[B64.charCodeAt(i)],b=_L[B64.charCodeAt(i+1)],c=_L[B64.charCodeAt(i+2)],e=_L[B64.charCodeAt(i+3)];U[p++]=a<<2|b>>4;if(p<_n)U[p++]=(b&15)<<4|c>>2;if(p<_n)U[p++]=(c&3)<<6|e}MT.setFashion(U);" # FMRAW
test=pre+"var D="+json.dumps(days,ensure_ascii=False)+""";var out=[];for(var k in D){var env=MT.newEnv();MT.resetFS();var errs=[],plots=0;var t0=Date.now();D[k].forEach(function(c,i){var r=MT.runCell(c,env);plots+=(r.plots||[]).length;if(!r.ok)errs.push(i+': '+r.text.split('\\n').pop().slice(0,110))});out.push('第'+k+'天 '+D[k].length+' 格，图 '+plots+' 张，报错 '+errs.length+'，用时 '+(Date.now()-t0)+'ms\\n  '+errs.join('\\n  '))}return out.join('\\n');"""
TM="MT.setStaticFile('../data/timemachine.txt',"+json.dumps(open(d+'timemachine.txt').read())+");"+"MT.setStaticFile('../data/airfoil_self_noise.dat',"+json.dumps(open(d+'airfoil_self_noise.dat').read())+");"
import tempfile;_tf=_os.path.join(tempfile.gettempdir(),'d2l_t4.js');open(_tf,'w').write('print((function(){'+js+TM+test+'})())')
r=subprocess.run(['/System/Library/Frameworks/JavaScriptCore.framework/Versions/Current/Helpers/jsc',_tf],capture_output=True,text=True);print(r.stdout,r.stderr[:500])
