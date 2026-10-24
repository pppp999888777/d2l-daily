
// ---------- torchvision：内嵌的 Fashion-MNIST 子集 ----------
let FM=null;const FM_N={train:6000,test:1000};
function setFashion(u8){const ntr=FM_N.train,nte=FM_N.test,P=784;let o=0;const take=n=>{const r=u8.subarray(o,o+n);o+=n;return r};FM={train:{imgs:take(ntr*P),labels:take(ntr)},test:{imgs:take(nte*P),labels:take(nte)}}}
function setFashionArrays(tr,trl,te,tel){FM_N.train=trl.length;FM_N.test=tel.length;FM={train:{imgs:Uint8Array.from(tr),labels:Uint8Array.from(trl)},test:{imgs:Uint8Array.from(te),labels:Uint8Array.from(tel)}}}
const tfObj=(kind,cfg)=>({t:'transform',kind,cfg:cfg||{}});
function tfList(t){if(!t||t.t==='none')return[];if(t.kind==='Compose')return t.cfg.list;return[t]}
function resizeImg(src,h,w,H,W){const out=new Float64Array(H*W),sy=h/H,sx=w/W;for(let i=0;i<H;i++){const fy=Math.min(h-1,Math.max(0,(i+.5)*sy-.5)),y0=Math.floor(fy),y1=Math.min(h-1,y0+1),dy=fy-y0;
  for(let j=0;j<W;j++){const fx=Math.min(w-1,Math.max(0,(j+.5)*sx-.5)),x0=Math.floor(fx),x1=Math.min(w-1,x0+1),dx=fx-x0;out[i*W+j]=(src[y0*w+x0]*(1-dx)+src[y0*w+x1]*dx)*(1-dy)+(src[y1*w+x0]*(1-dx)+src[y1*w+x1]*dx)*dy}}return out}
function fmItem(ds,i){const part=FM[ds.train?'train':'test'];if(i<0)i+=ds.n;if(i<0||i>=ds.n)err('IndexError',`index ${i} is out of range`);let px=Float64Array.from(part.imgs.subarray(i*784,i*784+784));let H=28,W=28,toT=false;
  for(const t of ds.tfs){if(t.kind==='Resize'){const s=t.cfg.size;px=resizeImg(px,H,W,s[0],s[1]);H=s[0];W=s[1]}else if(t.kind==='ToTensor'){toT=true}}
  return{px:toT?px.map(v=>v/255):px,H,W,toT,label:part.labels[i]}}
function fmGet(ds,i){const it=fmItem(ds,i);const img=it.toT?mkT([1,it.H,it.W],it.px,'float32'):{t:'pil',H:it.H,W:it.W,px:it.px};return tup([img,I(it.label)])}
function fmBatch(ds,idx){const f=fmItem(ds,idx[0]);const P=f.H*f.W,d=new Float64Array(idx.length*P),lab=[];idx.forEach((i,k)=>{const it=k?fmItem(ds,i):f;d.set(it.px,k*P);lab.push(it.label)});
  if(!f.toT)err('TypeError','default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class \'PIL.Image.Image\'>');return list([mkT([idx.length,1,f.H,f.W],d,'float32'),mkT([idx.length],lab,'int64')])}
function fmMake(a,kw){if(!FM)err('RuntimeError','Fashion-MNIST 数据还在解压，请过一两秒再运行这一格');const train=truthy(kwOr(kw,a,'train',1,B(true)));const tf=kwOr(kw,a,'transform',2,NONE);
  return{t:'fmnist',train,n:FM_N[train?'train':'test'],tfs:tfList(tf),tfRepr:tf.t==='none'?null:tf,root:str(kwOr(kw,a,'root',0,S('../data')))}}
function tfRepr(t){if(t.kind==='ToTensor')return'ToTensor()';if(t.kind==='Resize')return`Resize(size=${t.cfg.raw}, interpolation=bilinear, max_size=None, antialias=True)`;return'Compose(\n'+t.cfg.list.map(x=>'    '+tfRepr(x)).join('\n')+'\n)'}
function fmRepr(ds){return`Dataset FashionMNIST\n    Number of datapoints: ${ds.n}\n    Root location: ${ds.root}\n    Split: ${ds.train?'Train':'Test'}`+(ds.tfRepr?`\n    StandardTransform\nTransform: ${tfRepr(ds.tfRepr).replace(/\n/g,'\n           ')}`:'')}
const torchvision=mod('torchvision',{datasets:mod('torchvision.datasets',{FashionMNIST:fn('FashionMNIST',fmMake)}),
  transforms:mod('torchvision.transforms',{ToTensor:fn('ToTensor',()=>tfObj('ToTensor')),Resize:fn('Resize',a=>{const v=a[0];const s=v.t==='int'?[v.v,v.v]:iterate(v).map(intOf);return tfObj('Resize',{size:s,raw:v.t==='int'?String(v.v):`[${s.join(', ')}]`})}),Compose:fn('Compose',a=>tfObj('Compose',{list:iterate(a[0])}))})});

// ---------- 简化的 matplotlib：subplots / plot / imshow / display ----------
function mkAx(fig){return{t:'ax',fig,lines:[],image:null,xlabel:'',ylabel:'',title:'',legend:[],xlim:null,ylim:null,xscale:'linear',yscale:'linear',axisOff:false}}
function axAttr(ax,name){const toArr=v=>isT(v)?[...v.data]:iterate(v).map(x=>numOf(x));
  const M={plot:a=>{let x,y,fmt='-';const rest=a.filter(v=>v.t!=='str');const fs=a.find(v=>v.t==='str');if(fs)fmt=fs.v;if(rest.length>=2){x=toArr(rest[0]);y=toArr(rest[1])}else{y=toArr(rest[0]);x=y.map((_,i)=>i)}ax.lines.push({x,y,fmt});return list([])},
    cla:()=>{ax.lines=[];ax.image=null;return NONE},set_xlabel:a=>{ax.xlabel=a[0]&&a[0].t!=='none'?str(a[0]):'';return NONE},set_ylabel:a=>{ax.ylabel=a[0]&&a[0].t!=='none'?str(a[0]):'';return NONE},
    set_xlim:a=>{if(a[0]&&a[0].t!=='none')ax.xlim=iterate(a[0]).map(numOf);return NONE},set_ylim:a=>{if(a[0]&&a[0].t!=='none')ax.ylim=iterate(a[0]).map(numOf);return NONE},
    set_xscale:a=>{ax.xscale=str(a[0]);return NONE},set_yscale:a=>{ax.yscale=str(a[0]);return NONE},legend:a=>{if(a[0]&&a[0].t!=='none')ax.legend=iterate(a[0]).map(str);return NONE},grid:()=>NONE,
    set_title:a=>{ax.title=str(a[0]);return NONE},imshow:a=>{const v=a[0];if(v.t==='pil'){ax.image={H:v.H,W:v.W,px:Float64Array.from(v.px,p=>p/255)};return NONE}const T=asTensor(v);const s=T.shape.filter(d=>d!==1);if(s.length!==2)err('TypeError',`Invalid shape (${T.shape.join(', ')}) for image data`);ax.image={H:s[0],W:s[1],px:Float64Array.from(T.data)};return NONE},
    get_xaxis:()=>({t:'mod',name:'XAxis',attrs:{set_visible:fn('set_visible',()=>{ax.axisOff=true;return NONE})}}),get_yaxis:()=>({t:'mod',name:'YAxis',attrs:{set_visible:fn('set_visible',()=>{ax.axisOff=true;return NONE})}}),scatter:a=>{ax.lines.push({x:toArr(a[0]),y:toArr(a[1]),fmt:'o'});return NONE},axhline:()=>NONE};
  if(name==='axes')return ax;if(M[name])return fn(name,M[name]);return DUMMY}
function subplots(a,kw){const nr=intOf(kwOr(kw,a,'nrows',0,I(1))),nc=intOf(kwOr(kw,a,'ncols',1,I(1)));const fig={t:'fig',rows:nr,cols:nc,axes:[],id:newId()};for(let i=0;i<nr*nc;i++)fig.axes.push(mkAx(fig));CUR.figs.push(fig);
  const axes=nr*nc===1?fig.axes[0]:{t:'axarr',fig,rows:nr,cols:nc};return tup([fig,axes])}
function figSpec(fig){return{kind:'fig',rows:fig.rows,cols:fig.cols,axes:fig.axes.map(ax=>({lines:ax.lines.map(l=>({x:l.x.slice(),y:l.y.slice(),fmt:l.fmt})),image:ax.image,xlabel:ax.xlabel,ylabel:ax.ylabel,title:ax.title,legend:ax.legend.slice(),xlim:ax.xlim,ylim:ax.ylim,yscale:ax.yscale}))}}
const ipyDisplay=mod('IPython.display',{display:fn('display',(a,kw,ctx)=>{const v=a[0];if(v&&v.t==='fig'){v.shown=true}else if(v)ctx.outBuf+=repr(v)+'\n';return NONE}),clear_output:fn('clear_output',()=>NONE)});
function d2lSetAxes(a){const ax=a[0];if(ax.t!=='ax')return NONE;const[xl,yl,xlim,ylim,xs,ys,lg]=a.slice(1);const M=axAttr;M(ax,'set_xlabel').call([xl||NONE]);M(ax,'set_ylabel').call([yl||NONE]);M(ax,'set_xlim').call([xlim||NONE]);M(ax,'set_ylim').call([ylim||NONE]);if(ys&&ys.t==='str')ax.yscale=ys.v;if(lg&&lg.t!=='none')M(ax,'legend').call([lg]);return NONE}

// ---------- d2l 包里原书 #@save 的函数（用原书的 Python 代码，在本解释器里执行） ----------
const D2L_PRELUDE=`
import torch
import torchvision
from torchvision import transforms
from torch.utils import data
from IPython import display

def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

def get_dataloader_workers():
    """使用4个进程来读取数据"""
    return 4

def load_data_fashion_mnist(batch_size, resize=None):
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))

def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = d2l.argmax(y_hat, axis=1)
    cmp = d2l.astype(y_hat, y.dtype) == y
    return float(d2l.reduce_sum(d2l.astype(cmp, y.dtype)))

class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), d2l.size(y))
    return metric[0] / metric[1]

def train_epoch_ch3(net, train_iter, loss, updater):
    """训练模型一个迭代周期（定义见第3章）"""
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]

class Animator:
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """训练模型（定义见第3章）"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    # 原书是 train_loss < 0.5。本页只内嵌了 6000 张训练图（原书 60000 张），每个 epoch 更新次数少 10 倍，10 个 epoch 后损失约 0.59，所以阈值放宽到 0.65
    assert train_loss < 0.65, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

def predict_ch3(net, test_iter, n=6):
    """预测标签（定义见第3章）"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(d2l.argmax(net(X), axis=1))
    titles = [true +'\\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        d2l.reshape(X[0:n], (n, 28, 28)), 1, n, titles=titles[0:n])
`;
