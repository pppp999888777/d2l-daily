import os as _o0,base64 as _b0
def _rawb64():
    p=_o0.path.join(_o0.path.dirname(_o0.path.abspath(__file__)),'fmnist_raw_b64.txt')
    if not _o0.path.exists(p):
        import numpy as np
        z=np.load(p.replace('fmnist_raw_b64.txt','fmnist_subset.npz'))
        open(p,'w').write(_b0.b64encode(np.concatenate([z['Xtr'].ravel(),z['ytr'],z['Xte'].ravel(),z['yte']]).astype(np.uint8).tobytes()).decode())
    return open(p).read()
import json,sys,runpy,subprocess
import os as _os;d=_os.path.dirname(_os.path.abspath(__file__))+'/'
cellsfile,reffile=sys.argv[1],sys.argv[2]
C=runpy.run_path(d+cellsfile)['CELLS']
js=open(d+'minitorch2.js').read()
pre=''
if 'FMNIST' in sys.argv[4:]:
    pre="var B64='"+_rawb64()+"';var _A='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';var _L=new Uint8Array(256);for(var i=0;i<64;i++)_L[_A.charCodeAt(i)]=i;var _n=B64.length*3/4-(B64.endsWith('==')?2:B64.endsWith('=')?1:0);var U=new Uint8Array(_n);var p=0;for(var i=0;i<B64.length;i+=4){var a=_L[B64.charCodeAt(i)],b=_L[B64.charCodeAt(i+1)],c=_L[B64.charCodeAt(i+2)],e=_L[B64.charCodeAt(i+3)];U[p++]=a<<2|b>>4;if(p<_n)U[p++]=(b&15)<<4|c>>2;if(p<_n)U[p++]=(c&3)<<6|e}MT.setFashion(U);"
test=pre+"var CELLS="+json.dumps(C,ensure_ascii=False)+";var env=MT.newEnv();return JSON.stringify(CELLS.map(c=>MT.runCell(c,env).text));"
open(d+'t2.js','w').write('(function(){'+js+test+'})()')
raw=subprocess.run(['osascript','-l','JavaScript',d+'t2.js'],capture_output=True,text=True)
try: mine=json.loads(raw.stdout)
except Exception: print(raw.stdout[:3000],raw.stderr[:3000]); raise SystemExit
ref=[r.rstrip('\n') for r in json.load(open(d+reffile))]
skip=sys.argv[3].split(',') if len(sys.argv)>3 else []
bad=0
for i,(c,a,b) in enumerate(zip(C,mine,ref)):
    if any(s and s in c for s in skip): continue
    if a!=b: bad+=1;print('=== DIFF',i,':',c.replace('\n',' | ')[:200]);print('--- mine:\n'+a);print('--- real:\n'+b)
print('cells',len(C),'diffs',bad)
