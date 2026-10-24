import json,sys,io,contextlib,ast,os,runpy,traceback
import os as _os;d=_os.path.dirname(_os.path.abspath(__file__))+'/'
C=runpy.run_path(d+sys.argv[1])['CELLS'];out=[];g={}
import types
_d2l=types.ModuleType('d2l');_t=types.ModuleType('d2l.torch');_t.plot=lambda *a,**k:None;_t.set_figsize=lambda *a,**k:None;_t.use_svg_display=lambda *a,**k:None
import torch as _torch
for _n in ['ones','zeros','tensor','normal','matmul','arange','exp','sin','cos','float32','randn','eye','bmm','rand','meshgrid','cosh','sinh','sqrt']: setattr(_t,_n,getattr(_torch,_n))
_t.reshape=lambda x,s:x.reshape(s);_t.numpy=lambda x:x.detach().numpy();_t.plt=types.SimpleNamespace(scatter=lambda *a,**k:None,plot=lambda *a,**k:None,axhline=lambda *a,**k:None,legend=lambda *a,**k:None,gca=lambda:types.SimpleNamespace(set_xlabel=lambda *a:None,set_ylabel=lambda *a:None),rcParams={})
def _linreg(X,w,b): return _torch.matmul(X,w)+b
def _sq(y_hat,y): return (y_hat-y.reshape(y_hat.shape))**2/2
def _sgd(params,lr,batch_size):
    with _torch.no_grad():
        for p in params:
            p-=lr*p.grad/batch_size; p.grad.zero_()
def _synth(w,b,n):
    X=_torch.normal(0,1,(n,len(w)));y=_torch.matmul(X,w)+b;y+=_torch.normal(0,0.01,y.shape);return X,y.reshape((-1,1))
from torch.utils import data as _data
def _load_array(arrs,bs,is_train=True): return _data.DataLoader(_data.TensorDataset(*arrs),bs,shuffle=is_train)
_t.synthetic_data=_synth;_t.load_array=_load_array;_t.linreg=_linreg;_t.squared_loss=_sq;_t.sgd=_sgd
_d2l.torch=_t;sys.modules['d2l']=_d2l;sys.modules['d2l.torch']=_t;_mi=types.ModuleType('matplotlib_inline');_mi.backend_inline=types.SimpleNamespace(set_matplotlib_formats=lambda *a:None);sys.modules['matplotlib_inline']=_mi
if 'FMNIST' in sys.argv[3:]:
    import numpy as _np, torchvision as _tv, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as _plt
    from PIL import Image as _Image
    _z=_np.load(d+'fmnist_subset.npz')
    class _FM(_torch.utils.data.Dataset):
        def __init__(self, root, train=True, transform=None, download=False):
            self.X=_z['Xtr'] if train else _z['Xte']; self.y=_z['ytr'] if train else _z['yte']; self.transform=transform
        def __len__(self): return len(self.y)
        def __getitem__(self,i):
            img=_Image.fromarray(self.X[i].reshape(28,28),mode='L')
            return (self.transform(img) if self.transform else img), int(self.y[i])
    _tv.datasets.FashionMNIST=_FM
    _t.plt=_plt;_t.argmax=lambda x,*a,**k:x.argmax(*a,**k);_t.astype=lambda x,*a,**k:x.type(*a,**k);_t.reduce_sum=lambda x,*a,**k:x.sum(*a,**k);_t.size=lambda x,*a,**k:x.numel(*a,**k);_t.log=_torch.log
    def _set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
        axes.set_xlabel(xlabel); axes.set_ylabel(ylabel); axes.set_xscale(xscale); axes.set_yscale(yscale); axes.set_xlim(xlim); axes.set_ylim(ylim)
        if legend: axes.legend(legend)
        axes.grid()
    _t.set_axes=_set_axes
    import time as _time
    class _Timer:
        def __init__(self): self.times=[]; self.start()
        def start(self): self.tik=_time.time()
        def stop(self): self.times.append(_time.time()-self.tik); return self.times[-1]
        def avg(self): return sum(self.times)/len(self.times)
        def sum(self): return sum(self.times)
        def cumsum(self): return _np.array(self.times).cumsum().tolist()
    _t.Timer=_Timer
    import re as _re
    _js=open(d+'mt2_vision.js').read(); _pre=_js[_js.index('const D2L_PRELUDE=`')+19:_js.index('`;',_js.index('const D2L_PRELUDE=`'))].replace('\\\\n','\\n')
    _ip=types.ModuleType('IPython');_ip.get_ipython=lambda:None;_ip.version_info=(8,20,0);_t.d2l=_t;_DL=_torch.utils.data.DataLoader
    def _DL0(*a,**k):
        k['num_workers']=0;return _DL(*a,**k)
    _torch.utils.data.DataLoader=_DL0;_ip.display=types.SimpleNamespace(display=lambda *a,**k:None,clear_output=lambda *a,**k:None);sys.modules['IPython']=_ip;sys.modules['IPython.display']=_ip.display
    exec(_pre, _t.__dict__)
C=['\n'.join(l for l in c.split('\n') if not l.startswith('%')) for c in C]
os.makedirs(d+'ref2/work',exist_ok=True);os.makedirs(d+'ref2/data',exist_ok=True)
import shutil;shutil.copy(d+'timemachine.txt',d+'ref2/data/timemachine.txt');shutil.copy(d+'airfoil_self_noise.dat',d+'ref2/data/airfoil_self_noise.dat')  # 第 8 章：d2l.download 返回 ../data/timemachine.txt
os.chdir(d+'ref2/work')
for c in C:
    buf=io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            tree=ast.parse(c);last=None
            if tree.body and isinstance(tree.body[-1],ast.Expr): last=ast.Expression(tree.body.pop().value)
            exec(compile(tree,'<cell>','exec'),g)
            if last is not None:
                v=eval(compile(last,'<cell>','eval'),g)
                if v is not None and not c.rstrip().endswith(';'): print(repr(v))
        except Exception as e: print(f'{type(e).__name__}: {e}')
    out.append(buf.getvalue())
json.dump(out,open(d+sys.argv[2],'w'),ensure_ascii=False)
for c,o in zip(C,out): print('>>>',c.replace('\n',' | ')[:150]);print(o)
