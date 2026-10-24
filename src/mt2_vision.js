
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
    set_title:a=>{ax.title=str(a[0]);return NONE},imshow:(a,kw)=>{const v=a[0];const cm=kw&&kw.cmap?str(kw.cmap):null;if(v.t==='pil'){ax.image={H:v.H,W:v.W,px:Float64Array.from(v.px,p=>p/255)};return NONE}const T=asTensor(v);const s=T.shape.filter(d=>d!==1);if(s.length!==2)err('TypeError',`Invalid shape (${T.shape.join(', ')}) for image data`);ax.image={H:s[0],W:s[1],px:Float64Array.from(T.data),cmap:cm};return NONE},
    get_xaxis:()=>({t:'mod',name:'XAxis',attrs:{set_visible:fn('set_visible',()=>{ax.axisOff=true;return NONE})}}),get_yaxis:()=>({t:'mod',name:'YAxis',attrs:{set_visible:fn('set_visible',()=>{ax.axisOff=true;return NONE})}}),scatter:a=>{ax.lines.push({x:toArr(a[0]),y:toArr(a[1]),fmt:'o'});return NONE},axhline:()=>NONE};
  if(name==='axes')return ax;if(M[name])return fn(name,M[name]);return DUMMY}
function subplots(a,kw){const nr=intOf(kwOr(kw,a,'nrows',0,I(1))),nc=intOf(kwOr(kw,a,'ncols',1,I(1)));const fig={t:'fig',rows:nr,cols:nc,axes:[],id:newId()};for(let i=0;i<nr*nc;i++)fig.axes.push(mkAx(fig));CUR.figs.push(fig);
  if(kw.squeeze&&!truthy(kw.squeeze))return tup([fig,list(Array.from({length:nr},(_,r)=>list(fig.axes.slice(r*nc,r*nc+nc))))]);
  const axes=nr*nc===1?fig.axes[0]:{t:'axarr',fig,rows:nr,cols:nc};return tup([fig,axes])}
function figSpec(fig){return{kind:'fig',rows:fig.rows,cols:fig.cols,axes:fig.axes.map(ax=>({lines:ax.lines.map(l=>({x:l.x.slice(),y:l.y.slice(),fmt:l.fmt})),image:ax.image,xlabel:ax.xlabel,ylabel:ax.ylabel,title:ax.title,legend:ax.legend.slice(),xlim:ax.xlim,ylim:ax.ylim,xscale:ax.xscale,yscale:ax.yscale}))}}
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

def evaluate_loss(net, data_iter, loss):
    """评估给定数据集上模型的损失"""
    metric = d2l.Accumulator(2)
    for X, y in data_iter:
        out = net(X)
        y = d2l.reshape(y, out.shape)
        l = loss(out, y)
        metric.add(d2l.reduce_sum(l), d2l.size(l))
    return metric[0] / metric[1]

def predict_ch3(net, test_iter, n=6):
    """预测标签（定义见第3章）"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(d2l.argmax(net(X), axis=1))
    titles = [true +'\\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        d2l.reshape(X[0:n], (n, 28, 28)), 1, n, titles=titles[0:n])

# ---------- 第 8 章：文本（时光机器）----------
import collections
import re
import random
import os

DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'
DATA_HUB['time_machine'] = (DATA_URL + 'timemachine.txt',
                            '090b5e7e70c295757f55df93cb0a180b9691891a')

def download(name, cache_dir=os.path.join('..', 'data')):
    """本页不联网：数据已经放在网页里，直接返回文件路径"""
    url, sha1_hash = DATA_HUB[name]
    return os.path.join(cache_dir, url.split('/')[-1])

def read_time_machine():
    """将时间机器数据集加载到文本行的列表中"""
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

def tokenize(lines, token='word'):
    """将文本行拆分为单词或字符词元"""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)

class Vocab:
    """文本词表"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # 未知词元的索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs

def count_corpus(tokens):
    """统计词元的频率"""
    # 这里的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

def load_corpus_time_machine(max_tokens=-1):
    """返回时光机器数据集的词元索引列表和词表"""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    # 因为时光机器数据集中的每个文本行不一定是一个句子或一个段落，
    # 所以将所有文本行展平到一个列表中
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

def seq_data_iter_random(corpus, batch_size, num_steps):
    """使用随机抽样生成一个小批量子序列"""
    corpus = corpus[random.randint(0, num_steps - 1):]
    num_subseqs = (len(corpus) - 1) // num_steps
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    random.shuffle(initial_indices)

    def data(pos):
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield d2l.tensor(X), d2l.tensor(Y)

def seq_data_iter_sequential(corpus, batch_size, num_steps):
    """使用顺序分区生成一个小批量子序列"""
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = d2l.tensor(corpus[offset: offset + num_tokens])
    Ys = d2l.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y

class SeqDataLoader:
    """加载序列数据的迭代器"""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = d2l.seq_data_iter_random
        else:
            self.data_iter_fn = d2l.seq_data_iter_sequential
        self.corpus, self.vocab = d2l.load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)

def load_data_time_machine(batch_size, num_steps,
                           use_random_iter=False, max_tokens=10000):
    """返回时光机器数据集的迭代器和词表"""
    data_iter = SeqDataLoader(
        batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab

# ---------- 第 8 章：训练循环神经网络 ----------
import math
from torch import nn
from torch.nn import functional as F

def try_gpu(i=0):
    """本页没有 GPU，总是返回 CPU"""
    return torch.device('cpu')

class RNNModelScratch:
    """从零开始实现的循环神经网络模型"""
    def __init__(self, vocab_size, num_hiddens, device,
                 get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)

def predict_ch8(prefix, num_preds, net, vocab, device):
    """在prefix后面生成新字符"""
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: d2l.reshape(d2l.tensor(
        [outputs[-1]], device=device), (1, 1))
    for y in prefix[1:]:  # 预热期
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # 预测num_preds步
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])

def grad_clipping(net, theta):
    """裁剪梯度"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """训练网络一个迭代周期（定义见第8章）"""
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # 训练损失之和,词元数量
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # 在第一次迭代或使用随机抽样时初始化state
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # state对于nn.GRU是个张量
                state.detach_()
            else:
                # state对于nn.LSTM或对于我们从零开始实现的模型是个张量
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            # 因为已经调用了mean函数
            updater(batch_size=1)
        metric.add(l * d2l.size(y), d2l.size(y))
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()

def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    """训练模型（定义见第8章）"""
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    # 初始化
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # 训练和预测
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))

class RNNModel(nn.Module):
    """循环神经网络模型"""
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        # 如果RNN是双向的（之后将介绍），num_directions应该是2，否则应该是1
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        # 全连接层首先将Y的形状改为(时间步数*批量大小,隐藏单元数)
        # 它的输出形状是(时间步数*批量大小,词表大小)。
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # nn.GRU以张量作为隐状态
            return  torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens),
                                device=device)
        else:
            # nn.LSTM以元组作为隐状态
            return (torch.zeros((
                self.num_directions * self.rnn.num_layers,
                batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                        self.num_directions * self.rnn.num_layers,
                        batch_size, self.num_hiddens), device=device))

# ---------- 第 9、10 章：掩码和注意力 ----------
def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5),
                  cmap='Reds'):
    """显示矩阵热图"""
    d2l.use_svg_display()
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize,
                                 sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(d2l.numpy(matrix), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6);

def masked_softmax(X, valid_lens):
    """通过在最后一个轴上掩蔽元素来执行softmax操作"""
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
        X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                              value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)

class AdditiveAttention(nn.Module):
    """加性注意力"""
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)

class DotProductAttention(nn.Module):
    """缩放点积注意力"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)

def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算而变换形状"""
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    X = X.permute(0, 2, 1, 3)
    return X.reshape(-1, X.shape[2], X.shape[3])

def transpose_output(X, num_heads):
    """逆转transpose_qkv函数的操作"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)

class MultiHeadAttention(nn.Module):
    """多头注意力"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)
        output = self.attention(queries, keys, values, valid_lens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)

class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = d2l.zeros((1, max_len, num_hiddens))
        X = d2l.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

class Encoder(nn.Module):
    """编码器-解码器架构的基本编码器接口"""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError

class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络"""
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))

class AddNorm(nn.Module):
    """残差连接后进行层规范化"""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

class EncoderBlock(nn.Module):
    """Transformer编码器块"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout,
            use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(
            ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))
`;
