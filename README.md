# 深度学习自学站

跟着《动手学深度学习》第二版（PyTorch 版）每天学一点。书上的代码在网页里直接运行，不用注册、不用装软件，手机和电脑都能打开。

在线访问：https://pppp999888777.github.io/d2l-daily/

- `daily.html`：动手学深度学习每日课（代码格、张量图示、互动实验、课后题）
- `playground.html`：深度学习调试台（从机器学习到 Transformer 的总览）

## 工作原理

页面里的 `torch` 是用 JavaScript 写的简化实现（`src/mt2_*.js`），支持课程用到的 Python 语法、张量运算、自动求导、`nn`、`optim`、`DataLoader`、一个小型 pandas 和 matplotlib。每天的代码都先在真 PyTorch 2.14 / pandas 3.0 上运行，再和网页输出逐字对比（`src/cellsN.py`、`src/refN.json`）。

## 本地构建和测试

```bash
python3 src/build.py                 # 生成 daily.html、playground.html、index.html
python3 src/chksyntax.py             # 检查页面脚本有没有语法错误
python3 src/cmp.py cells7.py ref7.json "" FMNIST   # 和真 PyTorch 的输出对比（参照输出用 src/runref2.py 生成）
python3 src/runall.py                # 运行所有天的代码格
```

测试脚本用 macOS 自带的 JavaScriptCore（`jsc`）运行网页里的解释器。

## 版权说明

- 课程内容改编自《动手学深度学习》（阿斯顿·张、李沐、扎卡里·C. 立顿、亚历山大·J. 斯莫拉），原书以 Apache 2.0 许可证发布：https://zh.d2l.ai/
- 《时光机器》（H. G. Wells，1895）全文为公有领域作品，文本取自原书的数据仓库（d2l-data）。
- 翼型自噪声数据集（Airfoil Self-Noise）来自 UCI 机器学习库，CC BY 4.0 许可证：https://archive.ics.uci.edu/dataset/291/airfoil+self+noise
- Fashion-MNIST 数据集（训练集前 6000 张、测试集前 1000 张）来自 Zalando Research，MIT 许可证：https://github.com/zalandoresearch/fashion-mnist
