
// ---------- 课后题 ----------
// 题型：c 选择题；shape 写出代码结果的形状（网页现场运行判对错）；code 写代码，运行后用 check 检查
const QUIZ={
'1':[
 {t:'c',q:'机器学习的四个组件里，没有下面哪一个？',o:['数据','模型','目标函数','编译器'],a:3,why:'四个组件是数据、模型、目标函数（损失函数）和优化算法。编译器和机器学习无关。'},
 {t:'c',q:'模型在训练集上误差很小，在测试集上误差很大，这叫什么？',o:['欠拟合','过拟合','梯度消失','正则化'],a:1,why:'把训练数据的细节甚至噪声都记住了，换了新数据就不行，这是过拟合。两边误差都大才是欠拟合。'},
 {t:'shape',q:'下面代码结果的形状是什么？',code:'torch.arange(24).reshape(2, -1, 4)',why:'24 个元素，已知 2 和 4，-1 自动算成 24 ÷ (2×4) = 3。'},
 {t:'shape',q:'广播：下面相加的结果形状是什么？',code:'torch.arange(3).reshape(3, 1) + torch.arange(4).reshape(1, 4)',why:'(3, 1) 的列复制 4 份，(1, 4) 的行复制 3 份，都变成 (3, 4)。'},
 {t:'c',q:'X = torch.arange(12).reshape(3, 4)，X[1:3, 1] 一共选出几个元素？',o:['2 个','3 个','4 个','6 个'],a:0,why:'1:3 是第 1、2 行（不含 3），1 是第 1 列，所以是 2 行 × 1 列 = 2 个元素：5 和 9。'},
 {t:'c',q:'下面哪个会报错？',o:['torch.ones(2, 3) + torch.ones(3)','torch.ones(2, 3) + torch.ones(2)','torch.ones(2, 3) + torch.ones(1, 3)','torch.ones(2, 3) + torch.ones(2, 1)'],a:1,why:'广播从最后一个轴往前对齐。(2, 3) 和 (2,) 的最后一个轴是 3 对 2，既不相等也不是 1，所以报错。'},
 {t:'c',q:'Y = Y + X 和 Y += X 有什么区别？',o:['没有区别','前者结果更精确','前者分配新内存、Y 指向新地址；后者原地修改，地址不变','后者会报错'],a:2,why:'第 1 天 2.1.5 节用 id() 演示过。训练时更新上亿个参数，原地修改能省内存。'},
 {t:'code',q:'写代码：造一个形状为 (256, 100, 64) 的全 1 张量 z（256 个学生、100 步、每步 64 个数），再取出第 3 个学生（下标 2）第 10 步（下标 9）的那 64 个数，存成 v。',start:'z = \nv = \nv.shape',check:env=>{const z=env.vars.z,v=env.vars.v;if(!z||!MT.isT(z)||z.shape.join()!=='256,100,64')return'z 的形状应该是 (256, 100, 64)';if(!v||!MT.isT(v)||v.shape.join()!=='64')return'v 的形状应该是 (64,)，试试 z[2, 9]';return true},why:'z = torch.ones((256, 100, 64))，v = z[2, 9]。两个整数下标会把前两个轴去掉，剩下 64。'},
],
'2':[
 {t:'c',q:'data.isna().sum() 返回的是什么？',o:['表格总共有几格','每一列缺失值的个数','每一行缺失值的个数','缺失值的平均数'],a:1,why:'isna() 把每格变成是否缺失（True/False），sum() 默认按列加起来，得到每列缺几个。'},
 {t:'c',q:'Alley 列只有 "Pave" 和缺失两种值。pd.get_dummies(inputs, dummy_na=True) 会生成哪几列？',o:['Alley','Alley_Pave','Alley_Pave 和 Alley_nan','Alley_0 和 Alley_1'],a:2,why:'每个类别一列，dummy_na=True 再把缺失当成一个类别，所以是 Alley_Pave 和 Alley_nan。'},
 {t:'shape',q:'按轴求和后的形状？',code:'torch.arange(24).reshape(2, 3, 4).sum(axis=1)',why:'对哪个轴求和，形状里就删掉哪个数：(2, 3, 4) 删掉轴 1 的 3，得到 (2, 4)。'},
 {t:'shape',q:'加上 keepdims=True 呢？',code:'torch.arange(24).reshape(2, 3, 4).sum(axis=1, keepdims=True)',why:'keepdims=True 时被求和的轴不删，长度变成 1：(2, 1, 4)。'},
 {t:'shape',q:'矩阵乘法的结果形状？',code:'torch.ones(5, 4) @ torch.ones(4, 3)',why:'(5, 4) @ (4, 3)：左边列数 4 等于右边行数 4，结果是 (5, 3)。'},
 {t:'c',q:'A 的形状是 (5, 4)，B 的形状是 (4, 3)。A * B 会怎样？',o:['得到 (5, 3)','得到 (5, 4)','报错，因为 * 是按元素乘，形状要能广播','得到一个数'],a:2,why:'* 是按元素相乘（Hadamard 积），(5, 4) 和 (4, 3) 没法广播，会报错。矩阵乘法要用 @。'},
 {t:'c',q:'向量 (6, 8) 的 L2 范数和 L1 范数分别是多少？',o:['10 和 14','14 和 10','100 和 14','10 和 48'],a:0,why:'L2 = √(36 + 64) = 10；L1 = |6| + |8| = 14。'},
 {t:'c',q:'论文里说“隐藏维度 d = 64”，这里的“维度”是什么意思？',o:['张量有 64 个轴','每个向量有 64 个数','有 64 层','批量大小是 64'],a:1,why:'“向量的维度”指向量的长度。“张量的维度”才是指轴的个数，两个意思容易搞混。'},
],
'3':[
 {t:'c',q:'f(x) = x³ 在 x = 2 处的导数是多少？',o:['6','8','12','3'],a:2,why:'幂律：(xⁿ)′ = n·xⁿ⁻¹，所以 f′(x) = 3x²，x = 2 时是 12。'},
 {t:'c',q:'y = (2x + 1)³，用链式法则，x = 1 时 dy/dx 是多少？',o:['27','54','18','6'],a:1,why:'令 u = 2x + 1，dy/du = 3u² = 3×9 = 27，du/dx = 2，相乘得 54。'},
 {t:'c',q:'f(x₁, x₂) = x₁² + 3x₂² 在点 (1, 1) 的梯度是什么？',o:['(1, 3)','(2, 6)','(2, 3)','(4, 0)'],a:1,why:'∂f/∂x₁ = 2x₁ = 2，∂f/∂x₂ = 6x₂ = 6。往 (−2, −6) 方向走一小步，函数下降最快。'},
 {t:'c',q:'x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)；y = (x ** 2).sum()；y.backward() 以后，x.grad 是什么？',o:['tensor([1., 2., 3.])','tensor([2., 4., 6.])','tensor([1., 4., 9.])','tensor(14.)'],a:1,why:'y = x₀² + x₁² + x₂²，对每个 xᵢ 的偏导是 2xᵢ。'},
 {t:'c',q:'对两个不同的 y 连续调用 backward，中间不清零，x.grad 会怎样？',o:['只保留最后一次的梯度','两次梯度加在一起','报错','变成 0'],a:1,why:'PyTorch 默认累加梯度。所以训练循环每一步都要先清零，比如 optimizer.zero_grad()。'},
 {t:'c',q:'为什么手动更新参数时要写在 with torch.no_grad(): 里？',o:['让计算更精确','“更新参数”这个动作不该被记进计算图，而且不能原地修改需要梯度的叶子节点','为了清零梯度','为了让参数变成整数'],a:1,why:'不放在 no_grad 里，x -= 0.1 * x.grad 会直接报错：a leaf Variable that requires grad is being used in an in-place operation。'},
 {t:'c',q:'u = y.detach() 以后，u 和 y 有什么关系？',o:['完全一样，梯度会经过 u 传回去','数值一样，但和计算图断开，梯度不会经过 u 往回传','u 是 y 的副本，数值会变','u 是 y 的梯度'],a:1,why:'detach 返回一个数值相同但不在计算图里的张量，用来把某个中间结果当常数处理。'},
],
'4':[
 {t:'c',q:'公平骰子掷 6000 次，点数 1 大约会出现多少次？',o:['600','1000','6000','6'],a:1,why:'概率是 1/6，6000 × 1/6 = 1000。实际会在 1000 上下浮动，掷得越多，相对频率越接近 1/6（大数定律）。'},
 {t:'c',q:'贝叶斯定理 P(A | B) 等于什么？',o:['P(A) · P(B)','P(B | A) · P(A) / P(B)','P(A, B) · P(B)','P(B | A) / P(A)'],a:1,why:'由 P(A, B) = P(B | A)P(A) = P(A | B)P(B) 推出。'},
 {t:'c',q:'HIV 检测的例子：如果人群感染率更低，检测阳性的人真正感染的概率会怎样？',o:['更高','不变','更低','变成 99%'],a:2,why:'病越罕见，健康人的误报在阳性结果里占的比例越大，阳性的可信度就越低。'},
 {t:'c',q:'BKT 的 guess 参数是哪个条件概率？',o:['P(会 | 答对)','P(答对 | 不会)','P(答错 | 会)','P(学会)'],a:1,why:'guess 是不会但蒙对的概率；slip 是 P(答错 | 会)，会了但失误。'},
 {t:'c',q:'BKT 中 p(会) = 0.5，guess = 0.5，slip = 0。学生答对一题后（先不算 learn），p(会 | 答对) 是多少？',o:['0.5','0.6667','0.75','1'],a:1,why:'P(会 | 对) = (1 − 0)·0.5 / [(1 − 0)·0.5 + 0.5·0.5] = 0.5 / 0.75 = 2/3。蒙对的概率高，所以答对一题只能提高一点信心。'},
 {t:'c',q:'公平骰子点数的方差是多少？',o:['3.5','1.7078','2.9167','6'],a:2,why:'E[X] = 3.5，E[X²] = 91/6 ≈ 15.17，方差 = 15.17 − 12.25 ≈ 2.9167，标准差 ≈ 1.7078。'},
 {t:'c',q:'想知道 torch.ones 有哪些参数，在 Python 里最直接的办法是？',o:['dir(torch.ones)','help(torch.ones)','print(torch.ones)','torch.ones.shape'],a:1,why:'help() 显示函数的签名和说明文档；dir() 列出一个模块里有哪些东西。'},
],
'5':[
 {t:'shape',q:'线性回归的预测：X 是 1000 个样本、每个 2 个特征，w 是 (2, 1)，b 是 (1,)。ŷ 的形状？',code:'torch.ones(1000, 2) @ torch.ones(2, 1) + torch.zeros(1)',why:'(1000, 2) @ (2, 1) = (1000, 1)，加 b 用广播，形状不变。'},
 {t:'c',q:'1000 个样本、批量大小 10，一个 epoch 更新几次参数？',o:['10 次','100 次','1000 次','1 次'],a:1,why:'每个小批量更新一次，1000 ÷ 10 = 100 个小批量。'},
 {t:'c',q:'学习率设得太大，最可能出现什么情况？',o:['训练更快、结果更好','损失越来越大甚至变成 nan（发散）','模型变得更简单','没有影响'],a:1,why:'每一步跳过了最低点而且越跳越远，第 1、3、5 天的实验里都看到过。'},
 {t:'c',q:'最小化平方损失，等价于在什么噪声假设下做极大似然估计？',o:['均匀分布','正态（高斯）分布','伯努利分布','指数分布'],a:1,why:'把正态分布的密度代进负对数似然，去掉常数就是平方误差之和。'},
 {t:'shape',q:'squared_loss 如果不 reshape：y_hat 是 (10, 1)，y 是 (10,)，y_hat - y 的形状？',code:'torch.ones(10, 1) - torch.zeros(10)',why:'广播把它们扩展成 (10, 10)，不报错但损失完全错了。所以要先 y.reshape(y_hat.shape)。'},
 {t:'c',q:'线性回归解析解是哪个公式？',o:['w* = XᵀX y','w* = (XᵀX)⁻¹Xᵀy','w* = X⁻¹y','w* = Xᵀy / n'],a:1,why:'把 b 并进 w（X 加一列全 1），令损失对 w 的梯度为 0 解出。'},
 {t:'c',q:'你论文代码里 BS = 256，ASSISTments 的训练集有 3718 个序列窗口。一个 epoch 更新多少次参数？',o:['14 次','15 次','256 次','3718 次'],a:1,why:'3718 ÷ 256 ≈ 14.5，最后一批不满 256 也算一批，所以是 15 次（和第 5 天练习 7 一样）。'},
],
'6':[
 {t:'c',q:'nn.Linear(784, 10) 的 weight 形状是什么？',o:['(784, 10)','(10, 784)','(784,)','(10,)'],a:1,why:'nn.Linear 的权重是 [输出数, 输入数]，内部算 X @ weight.T + bias。'},
 {t:'c',q:'nn.MSELoss() 默认返回什么？',o:['每个样本的损失','所有样本损失的平均值','所有样本损失的总和','最大的那个损失'],a:1,why:'默认 reduction="mean"。要每个样本的损失用 reduction="none"，要总和用 "sum"。'},
 {t:'c',q:'softmax([0, 0, 0]) 是多少？',o:['[0, 0, 0]','[1/3, 1/3, 1/3]','[1, 0, 0]','[1, 1, 1]'],a:1,why:'exp(0) = 1，每个都是 1/(1+1+1)。logit 全相同时概率平均分配。'},
 {t:'c',q:'模型给正确类别的概率是 0.25，交叉熵损失是多少？',o:['0.25','0.75','约 1.386（即 ln 4）','4'],a:2,why:'交叉熵 = −ln(0.25) = ln 4 ≈ 1.386。'},
 {t:'c',q:'交叉熵损失对 logit 的梯度是什么？',o:['softmax(o) − y（预测概率减独热标签）','y − softmax(o)','softmax(o)','−log softmax(o)'],a:0,why:'这是 softmax 配交叉熵最好用的性质：预测越准，梯度越小。'},
 {t:'c',q:'两类时，logit z = 0，sigmoid(z) 和这时的 BCE（真实标签是 1）分别是多少？',o:['0 和 0','0.5 和约 0.693','1 和 0','0.5 和 0.5'],a:1,why:'sigmoid(0) = 0.5；BCE = −ln(0.5) = ln 2 ≈ 0.693。'},
 {t:'c',q:'ECE（期望校准误差）衡量的是什么？',o:['预测的排序能力','模型给出的置信度和实际正确率差多少','训练用了多久','模型的参数个数'],a:1,why:'模型说 70% 时，是不是真有 70% 答对。AUC 衡量排序能力，ECE 衡量校准。'},
 {t:'c',q:'训练循环里忘了写 trainer.zero_grad()，会发生什么？',o:['报错','梯度在每一步不断累加，更新越来越乱','训练更快','没有影响'],a:1,why:'梯度默认累加（第 3 天学过），每一步的梯度都把前面所有步的加了进去。'},
],
'7':[
 {t:'shape',q:'Fashion-MNIST 一个批量 256 张图的 X，形状是什么？（下面代码造一个同样形状的张量）',code:'torch.zeros(256, 1, 28, 28)',why:'[批量, 通道, 高, 宽] = [256, 1, 28, 28]。灰度图只有 1 个通道。'},
 {t:'c',q:'transforms.ToTensor() 之后，像素值的范围是？',o:['0 到 255','0 到 1','−1 到 1','任意实数'],a:1,why:'ToTensor 把 0–255 的整数像素除以 255，变成 0–1 之间的 32 位浮点数。'},
 {t:'c',q:'softmax 回归（784 个输入，10 个输出）一共多少个参数？',o:['784','7840','7850','7940'],a:2,why:'权重 784 × 10 = 7840，加 10 个偏置，共 7850。'},
 {t:'c',q:'3 个样本，预测概率最大的类别是 [2, 1, 0]，真实标签是 [2, 0, 0]，精度是多少？',o:['1/3','2/3','1','0'],a:1,why:'第 1 个和第 3 个对了，第 2 个错了，2/3。'},
 {t:'c',q:'为什么要把 logit（不是 softmax 之后的概率）直接传给 nn.CrossEntropyLoss？',o:['算得更快','数值更稳定：内部用 LogSumExp 技巧一起算 softmax 和 log，避免溢出和 log(0)','这样精度更高','必须这样，否则报错'],a:1,why:'按定义算，exp 可能溢出成 inf，概率可能下溢成 0 让 log 变成 −inf。'},
 {t:'c',q:'32 位浮点数里 exp(100) 是多少？',o:['约 2.7e43','inf（溢出）','0','nan'],a:1,why:'float32 最大约 3.4e38，exp(100) ≈ 2.7e43 超出了，变成 inf。'},
 {t:'c',q:'训练越久，训练精度一直上升，测试精度却开始下降，说明什么？',o:['欠拟合','过拟合，该提前停止或加正则化','学习率太小','数据有错'],a:1,why:'模型开始记住训练集的细节。你论文按验证集 BCE 选检查点，就是一种提前停止。'},
],
'8':[
 {t:'c',q:'两层全连接网络中间没有激活函数，它等价于什么？',o:['一个更强的非线性模型','一层线性（仿射）模型','一个 MLP','一个 RNN'],a:1,why:'线性函数套线性函数还是线性函数：W = W⁽¹⁾W⁽²⁾，b = b⁽¹⁾W⁽²⁾ + b⁽²⁾。'},
 {t:'c',q:'ReLU 在 x = −2 处的导数是多少？',o:['−2','0','1','0.5'],a:1,why:'x < 0 时 ReLU(x) = 0，导数是 0，梯度被挡住。'},
 {t:'c',q:'sigmoid 在 x = 0 处的导数是多少？',o:['0','0.25','0.5','1'],a:1,why:'sigmoid′(x) = s(1 − s)，x = 0 时 s = 0.5，导数 0.25，这是它的最大值。'},
 {t:'c',q:'tanh 的取值范围是？',o:['(0, 1)','(−1, 1)','[0, +∞)','(−∞, +∞)'],a:1,why:'tanh 像 sigmoid，但关于原点对称，范围是 −1 到 1。'},
 {t:'c',q:'MLP：784 → 256（ReLU）→ 10，一共多少个参数？',o:['203530','200704','7850','204800'],a:0,why:'784 × 256 + 256 + 256 × 10 + 10 = 200704 + 256 + 2560 + 10 = 203530。'},
 {t:'c',q:'simpleKT 的输出头 nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)) 有多少个参数？',o:['8192','8321','8256','193'],a:1,why:'128 × 64 + 64 = 8256，64 × 1 + 1 = 65，共 8321。ReLU 没有参数。'},
 {t:'shape',q:'nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU()) 作用在 64 张图上，输出形状？',code:'nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU())(torch.zeros(64, 1, 28, 28))',why:'Flatten 把 [64, 1, 28, 28] 变成 [64, 784]，全连接层变成 [64, 256]，ReLU 不改形状。'},
],
'9':[
 {t:'c',q:'选“隐藏层几层、学习率多大”这类超参数时，应该看哪个数据集上的表现？',o:['训练集','验证集','测试集','三个都看，取平均'],a:1,why:'测试集只能在最后看一次。用测试集挑超参数，它就等于参与了训练，报告出来的结果会偏乐观。'},
 {t:'c',q:'训练损失 0.62、测试损失 0.63，两条曲线都很早就不降了。最可能是什么情况？',o:['过拟合','欠拟合','正好','测试集泄漏'],a:1,why:'两种误差都高而且差不多，说明模型连训练数据都拟合不好，表达能力不够，是欠拟合。可以换更复杂的模型。'},
 {t:'c',q:'只有 100 个训练样本，用 19 阶多项式（20 个参数）去拟合 3 阶多项式生成的数据，会出现什么？',o:['欠拟合','过拟合：训练损失低、测试损失降不下去','两种损失都降到 0','报错'],a:1,why:'模型比数据复杂得多，会去拟合噪声，高次项的权重也不再是 0。这是今天 4.4.3 节最后一个实验。'},
 {t:'c',q:'损失加上 (λ/2)‖w‖² 以后，用学习率 η 做一步梯度下降，w 的更新是什么？',o:['w ← w − η∂L/∂w','w ← (1 − ηλ)w − η∂L/∂w','w ← (1 + ηλ)w − η∂L/∂w','w ← w − λ∂L/∂w'],a:1,why:'(λ/2)‖w‖² 的梯度是 λw，所以 w − η(∂L/∂w + λw) = (1 − ηλ)w − η∂L/∂w。每一步先把 w 缩小一点，所以叫权重衰减。'},
 {t:'c',q:'torch.optim.SGD([{"params": net[0].weight, "weight_decay": 3}, {"params": net[0].bias}], lr=0.003) 的意思是？',o:['权重和偏置都衰减 3','只有权重衰减，偏置不衰减','只有偏置衰减','学习率是 3'],a:1,why:'这是参数组写法：第一组单独设了 weight_decay，第二组没设，用默认值 0。你论文只给策略头 g 加 L2，是同一个思路。'},
 {t:'c',q:'Dropout 概率 p = 0.5，某个单元原来的输出是 4。训练时它要么变成多少？',o:['0 或 4','0 或 2','0 或 8','2 或 6'],a:2,why:'被丢掉是 0，留下来要除以 (1 − p) = 0.5，变成 8。这样期望值仍然是 0.5×0 + 0.5×8 = 4。'},
 {t:'c',q:'调用 net.eval() 之后，nn.Dropout 层会怎样？',o:['丢弃概率变成 1','什么也不做，原样输出','丢弃概率减半','报错'],a:1,why:'评估时要用上所有单元，所以 Dropout 直接输出输入。你论文的 train_kt 训练时调 model.train()，评估前调 model.eval()。'},
 {t:'c',q:'你论文 experiment.py 每个 epoch 在验证集上算 BCE，比之前更低就存下 state_dict，最后恢复最好的那份。这起什么作用？',o:['加快训练','用验证集做模型选择，相当于提前停止，防止用到过拟合以后的参数','增大学习率','让测试集变大'],a:1,why:'后面的 epoch 可能已经开始过拟合，按验证集挑出来的检查点泛化最好。全程没有用到测试集。'},
 {t:'code',q:'写代码：造一个 nn.Dropout(0.25) 层 drop，把它作用在 torch.ones(4000) 上，结果存成 y。检查：y 里只有 0 和 4/3 两种值，0 的比例在 0.2 到 0.3 之间。',start:'drop = \ny = \n(y == 0).float().mean()',check:env=>{const y=env.vars.y;if(!y||!MT.isT(y)||y.shape.join()!=='4000')return'y 应该是形状 (4000,) 的张量';const k=4/3;let z=0;for(const v of y.data){if(v===0)z++;else if(v===1)return'y 里有 1：Dropout 没起作用。是不是调了 drop.eval()？';else if(Math.abs(v-k)>1e-5)return`y 里有 ${v}，留下来的值应该是 1/(1 − 0.25) ≈ 1.3333`}const f=z/4000;if(f<0.2||f>0.3)return`0 的比例是 ${f.toFixed(3)}，应该在 0.25 附近。是不是用了 eval() 或者 p 写错了？`;return true},why:'drop = nn.Dropout(0.25)，y = drop(torch.ones(4000))。新建的层默认处在训练模式，所以会丢弃。'},
],
'10':[
 {t:'c',q:'sigmoid 函数导数的最大值是多少？在哪里取到？',o:['1，在 x = 0','0.5，在 x = 0','0.25，在 x = 0','0.25，在 x 很大时'],a:2,why:'σ′(x) = σ(x)(1 − σ(x))，x = 0 时 σ = 0.5，导数是 0.25。x 很大或很小时导数接近 0。'},
 {t:'c',q:'一个 20 层的网络，每层把梯度乘大约 0.25。传到第 1 层时梯度大约变成原来的多少？',o:['5 倍','0.25 × 20 = 5','0.25²⁰ ≈ 10⁻¹²','不变'],a:2,why:'链式法则是连乘，不是相加。0.25²⁰ ≈ 9×10⁻¹³，前面的层几乎学不到东西，这就是梯度消失。'},
 {t:'c',q:'把一个隐藏层的所有权重都初始化成同一个常数，会怎样？',o:['训练更快','所有隐藏单元算出相同的值、收到相同的梯度，永远一样，相当于只有 1 个单元','直接报错','梯度爆炸'],a:1,why:'这是对称性问题。随机初始化打破对称。你论文的策略头 g 初始化成 0 没问题，因为它是输出层，不是隐藏层。'},
 {t:'c',q:'一个全连接层 n_in = 300、n_out = 200。Xavier 初始化的权重方差应该是多少？',o:['1/300','2/500 = 0.004','2/300','6/500'],a:1,why:'σ² = 2/(n_in + n_out) = 2/500。用均匀分布时边界是 √(6/500) ≈ 0.11。'},
 {t:'c',q:'用真实照片训练猫狗分类器，上线后要分类卡通图片。这是哪种分布偏移？',o:['协变量偏移','标签偏移','概念偏移','没有偏移'],a:0,why:'输入 x 的分布变了（照片变卡通），但“图里是猫还是狗”这个规律 p(y | x) 没变。'},
 {t:'c',q:'协变量偏移下，重要性权重 β(xᵢ) 等于什么？',o:['p_S(x)/p_T(x)','p_T(x)/p_S(x)','p(y | x)','1/n'],a:1,why:'目标分布的密度除以源分布（训练数据）的密度。在目标分布里更常见的样本权重更大。'},
 {t:'c',q:'你论文 OPE 里的权重 w = π(a | s)/μ(a | s)，μ 指什么？',o:['新训练的出题策略','收集日志时实际使用的出题策略','学生的真实掌握度','学习率'],a:1,why:'数据是在 μ 下收集的，想评估的是新策略 π。和 β = p_T/p_S 一样：想要的分布除以数据实际来自的分布。'},
 {t:'c',q:'nn.utils.clip_grad_norm_(params, 5.0) 做了什么？',o:['把每个梯度元素截断到 [−5, 5]','如果所有梯度合起来的 L2 范数超过 5，就整体按比例缩小到 5，方向不变','把学习率设成 5','每 5 步更新一次'],a:1,why:'按范数裁剪保持方向，只缩短长度。逐元素截断的是 clip_grad_value_。你论文 train_kt 每一步都这样做，防止 LSTM 梯度爆炸。'},
 {t:'code',q:'写代码：把张量 w = torch.tensor([0.5, 3.0, 40.0, 8.0]) 截断到最大 20，存成 wc；再算有效样本量 ess = (wc.sum())**2 / (wc**2).sum()。',start:'w = torch.tensor([0.5, 3.0, 40.0, 8.0])\nwc = \ness = \ness',check:env=>{const w=env.vars.wc,e=env.vars.ess;if(!w||!MT.isT(w)||w.data.length!==4)return'wc 应该是 4 个数的张量';if(Math.abs(w.data[2]-20)>1e-6||Math.abs(w.data[0]-0.5)>1e-6)return`wc 是 [${[...w.data].join(', ')}]，应该是 [0.5, 3, 20, 8]。用 w.clamp(max=20)。`;const ev=e&&(MT.isT(e)?e.data[0]:e.v);if(ev==null||Math.abs(ev-31.5**2/473.25)>1e-3)return`ess 应该约等于 ${(31.5**2/473.25).toFixed(4)}`;return true},why:'wc = w.clamp(max=20) 得到 [0.5, 3, 20, 8]，ESS = 31.5² / 473.25 ≈ 2.10。4 个样本只相当于 2 个，因为权重集中在一个样本上。'},
],
'11':[
 {t:'c',q:'自己写一个 nn.Module 子类，__init__ 里第一件事应该做什么？',o:['定义 forward','调用 super().__init__()','初始化优化器','调用 backward()'],a:1,why:'父类的构造函数会准备好记录子块和参数的结构（比如 _modules）。不先调用，后面 self.hidden = nn.Linear(...) 就会报错。'},
 {t:'c',q:'net = MLP()，写 net(X) 时实际发生了什么？',o:['直接调用 __init__','调用 __call__，它再调用 forward','调用 backward','什么也不做'],a:1,why:'nn.Module 的 __call__ 会调用你写的 forward（还会处理钩子等）。所以平时写 net(X)，不要直接写 net.forward(X)。'},
 {t:'c',q:'你论文的 DKT 类没有写 forward，为什么还能训练？',o:['PyTorch 会自动生成 forward','训练代码直接调用 model.encode(...) 和 model.readout(...)，从没写 model(x)','DKT 不需要训练','forward 写在了父类 KTBase 里'],a:1,why:'只有写 model(x) 时才需要 forward。参数照样能被 model.parameters() 找到、被优化器更新。'},
 {t:'c',q:'nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1)) 的 state_dict 里有哪些键？',o:["'0.weight', '0.bias', '2.weight', '2.bias'","'0.weight', '0.bias', '1.weight', '1.bias'","'weight', 'bias'","'linear1', 'linear2'"],a:0,why:'名字是子块在 Sequential 里的下标。ReLU（下标 1）没有参数，所以没有 1.xxx。'},
 {t:'c',q:'shared = nn.Linear(8, 8) 在一个 Sequential 里出现两次。训练时 shared.weight.grad 是什么？',o:['只有第一次出现的梯度','只有第二次出现的梯度','两处的梯度加在一起','报错'],a:2,why:'两处用的是同一个参数对象，反向传播时梯度会累加到同一个 .grad 上。'},
 {t:'c',q:'DKT(K=145, d=64) 的 lstm.weight_ih_l0 形状是 (256, 64)。256 是怎么来的？',o:['2 × 128','4 × 64：LSTM 有 4 组门，每组 64 个','145 + 111','批量大小'],a:1,why:'LSTM 的输入门、遗忘门、候选记忆、输出门各需要一个 d × d 的矩阵，PyTorch 把 4 个拼在一起，变成 4d × d。第 14 天会详细讲。'},
 {t:'c',q:'best = net.state_dict() 之后继续训练，best 里的值会怎样？',o:['不变','跟着参数一起变，因为 state_dict 返回的张量和参数共用内存','变成 0','报错'],a:1,why:'所以你论文写的是 {k: v.detach().clone() for k, v in model.state_dict().items()}，复制一份独立的内存。'},
 {t:'c',q:'clone = MLP(); clone.load_state_dict(torch.load("mlp.params")) 为什么要先写 MLP()？',o:['文件里只有参数，没有网络结构，要先用同样的类建一个模型','为了随机初始化','load_state_dict 要求先训练','没必要，可以省略'],a:0,why:'state_dict 只是“名字 → 张量”的字典。网络怎么连接写在类的代码里，所以要先建同样结构的模型，再把参数填进去。'},
 {t:'code',q:'写代码：写一个类 Scale(nn.Module)，它有一个可训练参数 self.s（初始值是 torch.ones(1)），forward 返回 X * self.s。建一个 m = Scale()，最后一行写 m(torch.tensor([1.0, 2.0]))。',start:'class Scale(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.s = \n    def forward(self, X):\n        return \n\nm = Scale()\nm(torch.tensor([1.0, 2.0]))',check:env=>{const m=env.vars.m;if(!m||m.t!=='obj')return'还没有名为 m 的模型';const s=m.attrs.s;if(!s||!MT.isT(s)||!s.isParam)return'self.s 应该是 nn.Parameter(torch.ones(1))，这样它才是可训练参数';const r=MT.runCell('[n for n, _ in m.named_parameters()], m(torch.tensor([1.0, 2.0]))',env);if(!r.ok)return'运行 m(...) 出错了：'+r.text.split('\n').pop();if(!/'s'/.test(r.text)||!/tensor\(\[1\., 2\.\]/.test(r.text))return'm(torch.tensor([1.0, 2.0])) 应该得到 tensor([1., 2.], grad_fn=...)，现在是：'+r.text;return true},why:'self.s = nn.Parameter(torch.ones(1))；forward 里 return X * self.s。输出带 grad_fn=<MulBackward0>，说明 s 会被训练。'},
],
'12':[
 {t:'c',q:'自回归模型 xₜ ≈ f(x_{t−1}, …, x_{t−τ}) 和隐变量自回归模型最大的区别是？',o:['前者用神经网络，后者不用','前者只看最近 τ 步；后者用一个隐状态 hₜ 概括全部历史，每步更新','前者只能预测一步','没有区别'],a:1,why:'隐变量模型的 hₜ = g(h_{t−1}, x_{t−1}) 可以带着很久以前的信息。BKT 和 DKT 都是隐变量模型。'},
 {t:'c',q:'用自己的预测当输入一直往后预测（多步预测），为什么越来越不准？',o:['模型过拟合了','每一步的误差都会被带到下一步的输入里，误差不断累积','学习率太大','数据太少'],a:1,why:'单步预测每次都用真实数据当输入，多步预测用的是带误差的预测值。'},
 {t:'c',q:'词表 Vocab 里编号 0 留给什么？',o:['最常见的词','<unk>，表示词表里没有的未知词元','空格','句号'],a:1,why:'vocab["没见过的词"] 返回 0。按出现次数排序后，最常见的词元编号是 1。'},
 {t:'c',q:'你论文 DKT 有 K = 145 个知识点。学生在知识点 7 上答对了，输入嵌入层的编号是多少？',o:['7','152','8','14'],a:1,why:'sk + K * rs = 7 + 145 × 1 = 152。答错的话是 7。所以嵌入表有 2K = 290 行。'},
 {t:'c',q:'齐普夫定律说的是什么？',o:['词频和排名成正比','第 i 常见的词出现次数大致和 1/iᵅ 成正比，在双对数坐标上是一条直线','所有词出现次数差不多','词越长越常见'],a:1,why:'少数词极其常见，大量的词只出现一两次。'},
 {t:'c',q:'顺序分区相比随机抽样，好处是什么？',o:['训练更快','同一行里相邻小批量在原序列里是接着的，上一批的隐状态可以传给下一批','不需要标签','随机性更大'],a:1,why:'随机抽样时相邻小批量在原序列里不挨着，每批都要重新初始化隐状态。'},
 {t:'shape',q:'RNN 一步：批量 3，输入 1 维，隐藏 4 维。下面结果的形状是？',code:'X, W_xh = torch.ones(3, 1), torch.ones(1, 4)\nH, W_hh = torch.ones(3, 4), torch.ones(4, 4)\ntorch.tanh(X @ W_xh + H @ W_hh)',why:'(3, 1) @ (1, 4) 和 (3, 4) @ (4, 4) 都是 (3, 4)，加起来还是 (3, 4)，就是新的隐状态 Hₜ。'},
 {t:'c',q:'RNN 处理长度为 100 的序列和长度为 10 的序列，参数个数相比如何？',o:['长序列的参数多 10 倍','一样多，每个时间步共用同一组 W_xh、W_hh、b_h','长序列的参数少','取决于批量大小'],a:1,why:'这是 RNN 的关键：参数在时间步之间共享，所以能处理任意长度的序列。'},
 {t:'c',q:'一个预测对错的模型，每次都说答对概率 0.5。它的 BCE 和困惑度分别是多少？',o:['0.5 和 0.5','log 2 ≈ 0.693 和 2','1 和 e','0 和 1'],a:1,why:'每次 −log 0.5 = log 2，困惑度 = exp(log 2) = 2，相当于在两个选项里完全犹豫。'},
],
'13':[
 {t:'shape',q:'一个小批量 X 的形状是 (32, 35)（批量 32，时间步 35），词表大小 28。下面结果的形状是？',code:'X = torch.zeros(32, 35, dtype=torch.int64)\nF = torch.nn.functional\nF.one_hot(X.T, 28)',why:'先转置成 (35, 32)，独热编码再加一维 28：(35, 32, 28)。第一个维度是时间，方便 for X in inputs 一步步取。'},
 {t:'c',q:'独热向量 [0, 0, 1, 0] 乘一个 4 行的矩阵 W，结果是什么？',o:['W 的第 3 行','W 的第 3 列','全 0','W 的所有元素之和'],a:0,why:'只有第 3 个位置是 1，乘出来正好取出 W 的第 3 行。nn.Embedding 就是直接按编号取行，省掉乘 0 的计算。'},
 {t:'c',q:'所有参数的梯度合起来长度是 10，θ = 5。裁剪后梯度怎样？',o:['每个元素都截断到 [−5, 5]','整体乘 0.5，长度变成 5，方向不变','变成 0','不变'],a:1,why:'g ← min(1, θ/‖g‖)·g = 0.5·g。'},
 {t:'c',q:'顺序分区训练时，为什么每个小批量开始前要对隐状态做 detach_()？',o:['为了把隐状态清零','让隐状态的值保留，但把它和上一个小批量的计算图断开，反向传播不再往前传','为了加快前向计算','为了换设备'],a:1,why:'不断开的话，反向传播会一路传回第一个小批量，越来越慢，也容易梯度爆炸。这正是截断的 BPTT。'},
 {t:'c',q:'nn.RNN(28, 256) 处理形状 (35, 32, 28) 的输入，返回的 Y 是什么形状？',o:['(35, 32, 28)','(35, 32, 256)','(1, 32, 256)','(32, 256)'],a:1,why:'Y 是每个时间步的隐状态。nn.RNN 不含输出层，要自己接一个 nn.Linear(256, 28)。第二个返回值才是 (1, 32, 256) 的最后状态。'},
 {t:'c',q:'你论文 DKT 用 nn.LSTM(d, d, batch_first=True)。输入 x 的形状应该是？',o:['(时间步, 批量, d)','(批量, 时间步, d)','(批量, d)','(d, 批量, 时间步)'],a:1,why:'batch_first=True 时批量在前。嵌入层输出 (批量, 时间步, d)，直接送进 LSTM，不用转置。'},
 {t:'c',q:'困惑度从 28 降到 1，说明什么？',o:['模型从“28 个字符里瞎猜”变成“每次都确定地猜对”','模型变差了','词表变小了','训练时间变长了'],a:0,why:'困惑度 = exp(平均交叉熵)。均匀乱猜 28 个字符时是 28，完美预测时是 1。'},
 {t:'c',q:'沿时间往回传梯度时，每步乘一个约 0.5 的数。传 30 步后梯度变成多少？',o:['15','0.5 × 30','0.5³⁰ ≈ 10⁻⁹','不变'],a:2,why:'连乘。远处时间步几乎收不到梯度，学不到长距离依赖。LSTM 的门控就是为了缓解这个问题。'},
 {t:'code',q:'写代码：照 grad_clipping 的写法，把两个梯度 g1 = torch.tensor([3.0, 4.0]) 和 g2 = torch.tensor([12.0]) 当成一个整体裁剪到 θ = 1。算出 norm（总长度），把裁剪后的结果存成 c1 和 c2。',start:'g1 = torch.tensor([3.0, 4.0])\ng2 = torch.tensor([12.0])\ntheta = 1\nnorm = \nc1 = \nc2 = \nc1, c2',check:env=>{const n=env.vars.norm,c1=env.vars.c1,c2=env.vars.c2;const nv=n&&(MT.isT(n)?n.data[0]:n.v);if(nv==null||Math.abs(nv-13)>1e-4)return'norm 应该是 √(3² + 4² + 12²) = 13';if(!c1||!MT.isT(c1)||Math.abs(c1.data[0]-3/13)>1e-4||Math.abs(c1.data[1]-4/13)>1e-4)return'c1 应该是 g1 × θ / norm = [0.2308, 0.3077]';if(!c2||!MT.isT(c2)||Math.abs(c2.data[0]-12/13)>1e-4)return'c2 应该是 g2 × θ / norm = [0.9231]';return true},why:'norm = torch.sqrt((g1 ** 2).sum() + (g2 ** 2).sum()) = 13；c1 = g1 * theta / norm，c2 = g2 * theta / norm。裁剪后总长度正好是 1。'},
],
'14':[
 {t:'c',q:'GRU 的更新门 Zₜ 全都接近 1 时，这一步会怎样？',o:['隐状态几乎原样保留，这一步的输入基本被跳过','隐状态被清零','变成普通 RNN','输出全是 1'],a:0,why:'Hₜ = Zₜ ⊙ H_{t−1} + (1 − Zₜ) ⊙ H̃ₜ，Z ≈ 1 时 Hₜ ≈ H_{t−1}。'},
 {t:'c',q:'GRU 的重置门 Rₜ 接近 0 时，候选隐状态 H̃ₜ 主要取决于什么？',o:['上一步的隐状态','当前输入 Xₜ','输出层','什么都不取决'],a:1,why:'H̃ₜ = tanh(Xₜ W_xh + (Rₜ ⊙ H_{t−1}) W_hh + b_h)，R ≈ 0 时过去被“重置”掉了。'},
 {t:'c',q:'LSTM 里梯度沿记忆元往回传一步，乘的是什么？',o:['W_hh','遗忘门 Fₜ','输出门 Oₜ','tanh 的导数'],a:1,why:'Cₜ = Fₜ ⊙ C_{t−1} + Iₜ ⊙ C̃ₜ，所以 ∂Cₜ/∂C_{t−1} = Fₜ。F 接近 1 时梯度几乎不衰减。'},
 {t:'c',q:'LSTM 有几个状态需要从一个时间步传到下一个？',o:['1 个：隐状态','2 个：隐状态 H 和记忆元 C','3 个：三个门','4 个'],a:1,why:'所以 nn.LSTM 返回 (out, (h_n, c_n))，初始状态也要传一个元组。'},
 {t:'shape',q:'DKT：2 个学生，每人 50 步，d = 32。nn.LSTM(32, 32, batch_first=True) 的输出 h 的形状是？',code:'lstm = nn.LSTM(32, 32, batch_first=True)\nx = torch.zeros(2, 50, 32)\nh, _ = lstm(x)\nh',why:'batch_first=True：(批量, 时间步, 隐藏维) = (2, 50, 32)。'},
 {t:'shape',q:'论文的 readout：从 (2, 49, 5) 的 logit 里，按下一题的知识点挑出一个。结果的形状是？',code:'logits = torch.zeros(2, 49, 5)\nq = torch.zeros(2, 49, 1, dtype=torch.int64)\nlogits.gather(-1, q)',why:'gather 的结果和下标 q 形状一样：(2, 49, 1)。再 .squeeze(-1) 变成 (2, 49)，和对错 rs[:, 1:] 对齐算 BCE。'},
 {t:'c',q:'DKT 用第 t 步的隐状态去预测哪一题？',o:['第 t 题','第 t + 1 题','第 1 题','最后一题'],a:1,why:'h_t 已经看过第 t 题的对错，用它预测下一题。所以 qpos 是 0 到 T−2，目标是 rs[:, 1:]。'},
 {t:'c',q:'模拟数据里“知道真实掌握状态时的 AUC”为什么到不了 1？',o:['代码有错','即使知道会不会，蒙对（guess）和失误（slip）还是随机的','因为用了 Dropout','因为学生太少'],a:1,why:'这是数据本身的随机性，任何模型都无法消除。真实数据里也一样，所以 KT 的 AUC 一般在 0.7 到 0.85 之间。'},
 {t:'code',q:'写代码：logits = torch.tensor([[0.0, 2.0, -1.0], [1.0, 0.5, 3.0]])，每行是对 3 个知识点的 logit。下一题考的知识点是 q = [1, 2]，真实对错是 y = [1.0, 0.0]。用 gather 取出对应的 logit 存成 picked（形状 (2,)），再算 loss = F.binary_cross_entropy_with_logits(picked, y)。',start:'from torch.nn import functional as F\nlogits = torch.tensor([[0.0, 2.0, -1.0], [1.0, 0.5, 3.0]])\nq = torch.tensor([1, 2])\ny = torch.tensor([1.0, 0.0])\npicked = \nloss = \nloss',check:env=>{const p=env.vars.picked,l=env.vars.loss;if(!p||!MT.isT(p)||p.shape.join()!=='2'||Math.abs(p.data[0]-2)>1e-6||Math.abs(p.data[1]-3)>1e-6)return'picked 应该是 tensor([2., 3.])，形状 (2,)。提示：logits.gather(-1, q.unsqueeze(-1)).squeeze(-1)';const want=(Math.log(1+Math.exp(-2))+Math.log(1+Math.exp(3)))/2;if(!l||!MT.isT(l)||Math.abs(l.data[0]-want)>1e-4)return`loss 应该约等于 ${want.toFixed(4)}`;return true},why:'picked = logits.gather(-1, q.unsqueeze(-1)).squeeze(-1) 得到 [2., 3.]。第一题答对，logit 2 预测得不错；第二题答错，logit 3 却很有信心说会对，损失主要来自这里。'},
],
'15':[
 {t:'c',q:'注意力汇聚的输出是什么？',o:['查询本身','分数最高的那个键','值的加权平均，权重由查询和各个键的相似程度决定','所有键的平均'],a:2,why:'f(q) = Σ softmax(a(q, kᵢ)) vᵢ。'},
 {t:'c',q:'Nadaraya-Watson 核回归里，查询、键、值分别是什么？',o:['查询是测试点 x，键是训练点 xᵢ，值是训练点的 yᵢ','查询是 yᵢ，键是 xᵢ，值是 x','三者都是 x','没有键'],a:0,why:'用测试点和每个训练点的距离算权重，再对训练点的 y 加权平均。'},
 {t:'c',q:'核回归里 w 很小（核很宽）会怎样？',o:['预测贴着噪声走','几乎所有训练点权重相同，预测接近平均汇聚的那条平线','报错','预测等于真实函数'],a:1,why:'−((x − xᵢ)·w)²/2 几乎都是 0，softmax 后接近均匀分布。'},
 {t:'shape',q:'批量矩阵乘法的结果形状？',code:'torch.bmm(torch.ones(2, 1, 4), torch.ones(2, 4, 6))',why:'(2, 1, 4) 和 (2, 4, 6)：两个批量各自做 (1, 4) @ (4, 6)，得到 (2, 1, 6)。'},
 {t:'c',q:'masked_softmax 怎样让某些位置的权重变成 0？',o:['把权重直接设成 0 再归一化','把这些位置的分数换成 −10⁶，softmax 后 exp(−10⁶) ≈ 0','删掉这些位置','乘以 0.5'],a:1,why:'这样剩下位置的权重仍然加起来等于 1。'},
 {t:'c',q:'缩放点积注意力为什么要除以 √d？',o:['让计算更快','q 和 k 的分量方差为 1 时，点积的方差是 d；除以 √d 把分数的方差拉回 1，避免 softmax 饱和','为了让权重和为 1','为了防止过拟合'],a:1,why:'d 大时不缩放的分数很大，softmax 几乎只选一个键，梯度接近 0。'},
 {t:'c',q:'查询和键的长度不同（比如查询 20 维、键 2 维），该用哪种评分函数？',o:['缩放点积','加性注意力：先各自用线性层变到同一长度','都不行','点积不缩放'],a:1,why:'点积要求两个向量一样长。加性注意力的 W_q、W_k 把它们映射到同一个 h 维空间。'},
 {t:'c',q:'你论文 DKVMN 读记忆时 w = softmax(qe @ Mk.t())，这里的查询是什么？',o:['学生上一题的对错','下一题知识点的嵌入 qe','记忆矩阵 M','输出层'],a:1,why:'qe 和 20 个记忆槽的键 Mk 做点积，softmax 后的权重用来对记忆 M（值）加权求和。'},
 {t:'code',q:'写代码：一个查询 q = torch.tensor([1.0, 0.0])，三个键 k = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])，三个值 v = torch.tensor([10.0, 20.0, 30.0])。用缩放点积算权重 w（softmax(k @ q / √2)），再算输出 out = (w * v).sum()。',start:'import math\nq = torch.tensor([1.0, 0.0])\nk = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])\nv = torch.tensor([10.0, 20.0, 30.0])\nw = \nout = \nw, out',check:env=>{const w=env.vars.w,o=env.vars.out;const s=[1,0,1].map(x=>x/Math.SQRT2),e=s.map(Math.exp),z=e.reduce((a,b)=>a+b,0),ww=e.map(x=>x/z);if(!w||!MT.isT(w)||w.data.length!==3||ww.some((x,i)=>Math.abs(x-w.data[i])>1e-4))return`w 应该约等于 [${ww.map(x=>x.toFixed(4)).join(', ')}]。提示：torch.softmax(k @ q / math.sqrt(2), dim=0)`;const oo=ww[0]*10+ww[1]*20+ww[2]*30;if(!o||!MT.isT(o)||Math.abs(o.data[0]-oo)>1e-3)return`out 应该约等于 ${oo.toFixed(4)}`;return true},why:'分数是 [1, 0, 1] / √2，第 1 和第 3 个键和查询一样相似，权重相同且比第 2 个大。输出约 19.9，偏向 10 和 30 的平均值。'},
],
'16':[
 {t:'c',q:'num_hiddens = 64、num_heads = 2 的多头注意力，每个头的查询和键是几维？',o:['64','32','128','2'],a:1,why:'每个头 num_hiddens / num_heads = 32 维。你论文 causal_attn 里的 dh = d // heads 就是这个。'},
 {t:'shape',q:'transpose_qkv 之后的形状？（批量 2，4 个位置，num_hiddens 8，2 个头）',code:'X = torch.zeros(2, 4, 8)\nX = X.reshape(2, 4, 2, -1).permute(0, 2, 1, 3)\nX.reshape(-1, X.shape[2], X.shape[3])',why:'(2, 4, 8) → (2, 4, 2, 4) → 交换后 (2, 2, 4, 4) → 把头并进批量 (4, 4, 4)。'},
 {t:'c',q:'自注意力里，查询、键、值来自哪里？',o:['三个不同的序列','同一个序列（各自再经过线性变换）','查询来自序列，键值是参数','只有值来自序列'],a:1,why:'所以叫“自”注意力。SAKT 稍有不同：查询来自下一题的知识点，键值来自做过的题。'},
 {t:'c',q:'和 RNN 比，自注意力最大的优点是？',o:['参数更少','所有位置可以并行计算，任意两个位置一步就能连上','计算量随长度线性增长','不需要训练'],a:1,why:'代价是计算量和序列长度的平方成正比。'},
 {t:'c',q:'为什么自注意力需要位置编码？',o:['让计算更快','自注意力本身看不出先后顺序，打乱顺序结果只是跟着换位置','为了归一化','为了加掩码'],a:1,why:'加上位置编码后，同一道题出现在第 3 步和第 90 步，表示就不一样了。'},
 {t:'c',q:'你论文的 SAKT 用的是哪种位置编码？',o:['原书的正弦位置编码','可学习的位置嵌入 nn.Embedding(maxlen, d)','没有位置编码','二进制编码'],a:1,why:'self.pos = nn.Embedding(maxlen, d)，每个位置一个可训练的向量。两种做法都很常见。'},
 {t:'c',q:'因果掩码 torch.triu(ones, diagonal=1) 把哪些位置遮住？',o:['对角线','对角线以下（过去）','对角线以上（未来）：第 i 个查询看不到第 j > i 个键','全部'],a:2,why:'预测第 t+1 题时只能用第 t 题及以前的记录，否则就是偷看答案。'},
 {t:'c',q:'你论文 causal_attn 里 s.masked_fill((dist < 0)..., -1e9)，dist < 0 表示什么？',o:['键在查询之前','键的位置在查询之后，也就是未来','两题知识点不同','学生答错了'],a:1,why:'dist = qpos − jpos，小于 0 说明键的位置比查询还靠后。'},
 {t:'code',q:'写代码：造一个 4 × 4 的因果掩码 mask（对角线以上为 True），把全 0 的分数矩阵 s = torch.zeros(4, 4) 中被掩码的位置填成 -1e9，softmax 后存成 w（按最后一维）。',start:'s = torch.zeros(4, 4)\nmask = \nw = \nw',check:env=>{const w=env.vars.w;if(!w||!MT.isT(w)||w.shape.join()!=='4,4')return'w 应该是 4 × 4 的张量';const want=[[1,0,0,0],[.5,.5,0,0],[1/3,1/3,1/3,0],[.25,.25,.25,.25]];for(let i=0;i<4;i++)for(let j=0;j<4;j++)if(Math.abs(w.data[i*4+j]-want[i][j])>1e-4)return`第 ${i+1} 行应该是 [${want[i].map(x=>x.toFixed(2)).join(', ')}]`;return true},why:'mask = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)；w = torch.softmax(s.masked_fill(mask, -1e9), dim=-1)。第 i 行把权重平分给前 i 个位置。'},
],
'17':[
 {t:'c',q:'Transformer 编码器块里，每个子层外面包的是什么？',o:['池化层','残差连接 + 层规范化：LayerNorm(X + Dropout(子层(X)))','另一个注意力','卷积'],a:1,why:'原书的 AddNorm。SAKT 里的 ln1(drop(Wo(o)) + Qe) 和 ln2(o + drop(ffn(o))) 都是这个结构。'},
 {t:'c',q:'层规范化（LayerNorm）对什么做标准化？',o:['每个特征在整个批量上','每个样本（每个位置）自己的所有特征','所有参数','学习率'],a:1,why:'批量规范化才是按特征、跨样本。层规范化不依赖批量大小，适合序列模型。'},
 {t:'shape',q:'基于位置的前馈网络 PositionWiseFFN(4, 8, 4) 作用在形状 (2, 5, 4) 的输入上，输出形状？',code:'ffn = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))\nffn(torch.ones(2, 5, 4))',why:'线性层只作用在最后一维，每个位置单独算：(2, 5, 4)。'},
 {t:'c',q:'残差连接 Y = X + f(X) 对训练有什么好处？',o:['减少参数','反向传播时梯度可以沿着“+ X”直接传回去，深层网络不容易梯度消失','防止过拟合','让输出变成 0 到 1'],a:1,why:'∂Y/∂X = I + ∂f/∂X，至少有一个单位矩阵，梯度不会被连乘到 0。'},
 {t:'c',q:'TransformerEncoder 里为什么嵌入要乘 √num_hiddens 再加位置编码？',o:['为了归一化','位置编码的值在 −1 到 1 之间，嵌入的值较小，乘一下让两者大小相当','为了加快训练','没有原因'],a:1,why:'否则位置信息会盖过词本身的信息。'},
 {t:'c',q:'SAKT 的 ln1(drop(Wo(o)) + Qe) 里，残差加的是什么？',o:['做过的题的嵌入','下一题（查询）的嵌入 Qe','位置编码','0'],a:1,why:'SAKT 的输出位置对应的是“要预测的题”，所以残差连接回到查询本身。'},
 {t:'c',q:'AKT 的 causal_attn 里 s = s − decay × dist 起什么作用？',o:['防止梯度爆炸','离得越远的题注意力分数越低，模拟遗忘','因果掩码','归一化'],a:1,why:'decay = softplus(th1) 是可学习的，每个头一个。dist = 0 的最近一题不受影响。'},
 {t:'c',q:'在你论文的四个数据集上，哪个模型的预测 AUC 最常排第一？',o:['SAKT','AKT','DKT（四个里三个第一）','simpleKT'],a:2,why:'ASSISTments 2009、EdNet、Algebra 2005 上 DKT 最高，Junyi 上 AKT 略高。注意力模型不一定更准。'},
 {t:'code',q:'写代码：用 nn.LayerNorm(3) 规范化 x = torch.tensor([[1.0, 2.0, 3.0], [10.0, 10.0, 40.0]])，结果存成 y。检查：y 的每一行均值约为 0。',start:'x = torch.tensor([[1.0, 2.0, 3.0], [10.0, 10.0, 40.0]])\nln = \ny = \ny, y.mean(dim=1)',check:env=>{const y=env.vars.y;if(!y||!MT.isT(y)||y.shape.join()!=='2,3')return'y 应该是形状 (2, 3) 的张量';for(let i=0;i<2;i++){const r=[0,1,2].map(j=>y.data[i*3+j]),m=r.reduce((a,b)=>a+b,0)/3,v=r.reduce((a,b)=>a+(b-m)**2,0)/3;if(Math.abs(m)>1e-4||Math.abs(v-1)>1e-3)return`第 ${i+1} 行的均值应该是 0、方差应该约为 1，现在是 ${m.toFixed(4)} 和 ${v.toFixed(4)}`}if(Math.abs(y.data[0]+1.2247)>1e-3)return'第一行应该约为 [-1.2247, 0, 1.2247]';return true},why:'ln = nn.LayerNorm(3)；y = ln(x)。两行原来大小差很多，规范化后都变成均值 0、方差 1。第一行是 [−1.2247, 0, 1.2247]。'},
],
'18':[
 {t:'c',q:'随机梯度 ∇fᵢ(x)（随机挑一个样本 i）和真梯度 ∇f(x) 是什么关系？',o:['完全相等','期望相等（无偏），但单次有噪声','总是更大','没有关系'],a:1,why:'E[∇fᵢ(x)] = (1/n) Σ ∇fᵢ(x) = ∇f(x)。噪声让它在最优点附近停不下来，所以学习率要逐渐变小。'},
 {t:'c',q:'固定学习率的随机梯度下降，到最优点附近会怎样？',o:['精确停在最优点','在最优点附近不停地跳来跳去','发散','速度变快'],a:1,why:'梯度的噪声不会消失。让学习率随时间衰减（多项式衰减效果较好）可以解决。'},
 {t:'c',q:'为什么实践中用小批量，而不是一次一个样本？',o:['小批量更省内存','一次算一块矩阵比逐个算快得多（向量化），多个样本平均还能降低梯度噪声','一个样本无法求导','规定如此'],a:1,why:'原书 11.5.1 节的实验：一次性矩阵乘法比逐元素算快很多倍。'},
 {t:'c',q:'动量法 vₜ = βv_{t−1} + gₜ，在梯度正负交替的方向上会怎样？',o:['越走越快','正负抵消，震荡变小','不变','发散'],a:1,why:'而在梯度一直同号的方向上会越积越大，走得更快。'},
 {t:'c',q:'β = 0.9 时，动量大约相当于对过去多少步的梯度做平均？',o:['1 步','约 10 步（1/(1 − β)）','90 步','无穷多步'],a:1,why:'权重 β^τ 之和是 1/(1 − β) = 10。β = 0.99 时约 100 步。'},
 {t:'c',q:'Adam 里为什么要做偏差修正 v̂ = v / (1 − β₁ᵗ)？',o:['防止梯度爆炸','v 和 s 从 0 开始，前几步会偏小，除以 1 − βᵗ 把它们放大回正确的量级','为了加速','为了正则化'],a:1,why:'例如 t = 1 时 v₁ = 0.1 g₁，除以 1 − 0.9 = 0.1 后正好是 g₁。'},
 {t:'c',q:'Adam 的更新 η·v̂/(√ŝ + ε) 中，除以 √ŝ 起什么作用？',o:['让学习率变大','按每个参数梯度的典型大小归一化：梯度大的参数步子缩小，梯度小的放大','防止过拟合','让更新变成整数'],a:1,why:'所以每个参数每步大约走 η 那么远，对学习率没那么敏感。'},
 {t:'c',q:'你论文第 1349 行 torch.optim.Adam(params, lr=lr)，DKT 用的 β₁、β₂ 是多少？',o:['0.9 和 0.999（默认值）','0.5 和 0.9','0 和 0','没有用 Adam'],a:0,why:'只传了 lr，其余都是 PyTorch 默认：β₁ = 0.9，β₂ = 0.999，ε = 10⁻⁸。'},
 {t:'code',q:'写代码：手动做 2 步 Adam。参数 x = 1.0，每一步的梯度都是 g = 2x（也就是 f = x²），lr = 0.1，β₁ = 0.9，β₂ = 0.999，ε = 1e-8，v 和 s 从 0 开始，带偏差修正。两步之后的 x 存成 x。',start:'x = 1.0\nv, s = 0.0, 0.0\nb1, b2, lr, eps = 0.9, 0.999, 0.1, 1e-8\nfor t in range(1, 3):\n    g = 2 * x\n    # 在这里更新 v、s、x\n    \nx',check:env=>{let x=1,v=0,s=0;for(let t=1;t<3;t++){const g=2*x;v=0.9*v+0.1*g;s=0.999*s+0.001*g*g;x-=0.1*(v/(1-0.9**t))/(Math.sqrt(s/(1-0.999**t))+1e-8)}const got=env.vars.x;const gv=got&&(MT.isT(got)?got.data[0]:got.v);if(gv==null||Math.abs(gv-x)>1e-4)return`两步之后 x 应该约等于 ${x.toFixed(5)}，现在是 ${gv}`;return true},why:'v = b1*v + (1-b1)*g；s = b2*s + (1-b2)*g*g；x -= lr * (v/(1-b1**t)) / ((s/(1-b2**t)) ** 0.5 + eps)。第一步正好走了 lr = 0.1（v̂/√ŝ = g/|g| = 1），第二步也接近 0.1。'},
],
'19':[
 {t:'c',q:'日志由旧策略 μ 收集。直接算日志里的平均收益，估计的是什么？',o:['新策略 π 的价值','旧策略 μ 的价值','两者的平均','什么都不是'],a:1,why:'日志里的动作是按 μ 选的，平均收益自然代表 μ。'},
 {t:'c',q:'IPS 的权重 w = π(a | x) / μ(a | x)。某条记录 π(a | x) = 0.6，μ(a | x) = 0.05，权重是多少？',o:['0.03','12','0.55','1'],a:1,why:'0.6 / 0.05 = 12。旧策略很少选、新策略常选的动作，权重很大，IPS 的方差就来自这里。'},
 {t:'c',q:'SNIPS 和 IPS 的区别是？',o:['SNIPS 不用权重','SNIPS 用 Σwᵢ 代替 n 做分母：有一点偏差，但方差小很多','SNIPS 需要结果模型','没有区别'],a:1,why:'V̂ = Σ wᵢ rᵢ / Σ wᵢ。'},
 {t:'c',q:'DM（直接法）的结果模型 Q̂ 有系统偏差时，DM 估计会怎样？',o:['依然无偏','有偏差，而且样本再多也消除不了','方差变大','变成 IPS'],a:1,why:'DM 完全相信模型。今天的实验里 Q_bad 让 DM 偏了 −0.023，但标准差很小。'},
 {t:'c',q:'DR 为什么叫“双重稳健”？',o:['算了两遍','μ（权重）对或者 Q̂（结果模型）对，只要有一个对就无偏','用了两个模型取平均','有两个超参数'],a:1,why:'V̂ = Σ π Q̂ + w (r − Q̂(a))：Q̂ 错了，加权残差把它修回来；Q̂ 对了，残差期望为 0。'},
 {t:'c',q:'把权重截断到一个较小的上限（比如 5），会怎样？',o:['偏差和方差都变小','方差变小，但引入偏差','偏差变小，方差变大','没影响'],a:1,why:'今天的实验：截断到 5 时 IPS 的标准差从 0.020 降到 0.013，偏差从 +0.002 变成 −0.019。论文用的上限是 20。'},
 {t:'c',q:'归一化的有效样本量 ESS = (Σw)² / Σw² / n 接近 0 说明什么？',o:['估计很准','权重集中在极少数记录上，估计实际上只靠几条数据，很不稳定','样本太多','μ 和 π 一样'],a:1,why:'所有权重相等时 ESS = 1。新旧策略差得越远，ESS 越小。'},
 {t:'c',q:'你论文里 μ（旧策略）是怎么得到的？',o:['日志里直接记录了','按“上一题的知识点 → 这一题的知识点”的转移次数加平滑估计，并做交叉拟合','用神经网络学的','假设是均匀分布'],a:1,why:'prep_world 第 668 到 673 行。交叉拟合：一折学生用的 μ 在另一折上估计。'},
 {t:'code',q:'写代码：三条日志，w = torch.tensor([0.5, 2.0, 4.0])（已经算好的权重），r = torch.tensor([1.0, 0.0, 0.5])，结果模型对日志动作的预测 qa = torch.tensor([0.6, 0.2, 0.4])，对新策略的期望 dm = torch.tensor([0.5, 0.3, 0.45])。算出 ips、snips、dr（都是 0 维张量或数字）。',start:'w = torch.tensor([0.5, 2.0, 4.0])\nr = torch.tensor([1.0, 0.0, 0.5])\nqa = torch.tensor([0.6, 0.2, 0.4])\ndm = torch.tensor([0.5, 0.3, 0.45])\nips = \nsnips = \ndr = \nips, snips, dr',check:env=>{const v=n=>{const x=env.vars[n];return x==null?null:MT.isT(x)?x.data[0]:x.v};const want={ips:(0.5+0+2)/3,snips:2.5/6.5,dr:((0.5+0.5*0.4)+(0.3+2*(-0.2))+(0.45+4*0.1))/3};for(const k of['ips','snips','dr']){const g=v(k);if(g==null||Math.abs(g-want[k])>1e-4)return`${k} 应该约等于 ${want[k].toFixed(4)}`}return true},why:'ips = (w * r).mean() = 0.8333；snips = (w * r).sum() / w.sum() = 0.3846；dr = (dm + w * (r - qa)).mean() = 0.4833。'},
],
'20':[
 {t:'c',q:'你论文的实验分两条线，模拟器那条线最主要的作用是？',o:['让训练更快','能算出“真实的学习收益”（NLG 等）和 OPE 估计的真实值，这在真实日志里做不到','代替真实数据集','生成更多训练数据'],a:1,why:'真实日志里无法让同一个学生按新策略再学一遍；模拟器可以，所以能给出真值。'},
 {t:'c',q:'Safe-DFKT 是怎么训练出来的？',o:['从零开始用决策损失训练','先训练普通 DKT，再加策略头 g，用决策损失以 lr = 1e-3 微调','只训练策略头','用 SAKT 初始化'],a:1,why:'build_and_train：ensure_base 拿到训练好的 DKT，load_state_dict(..., strict=False)，再 train_kt(m, W, E_FT, 1e-3, cfg, ...)。'},
 {t:'c',q:'决策损失里的 −λ · V_DR 这一项，为什么是负号？',o:['写错了','最小化损失等价于最大化 DR 估计的出题价值','为了正则化','让 BCE 变小'],a:1,why:'优化器只会“最小化”。check_dr_identity 专门在运行时检查这个正负号。'},
 {t:'c',q:'安全屏障 γ · ReLU(BCE − (1 + ε) · BCE_base) 什么时候不起作用？',o:['总是起作用','BCE 不比原始 DKT 的 BCE 差超过 ε（0.5%）时，这一项等于 0','γ = 10 时','训练开始时'],a:1,why:'ReLU 让它只在预测明显变差时才惩罚，保证决策导向不以牺牲预测为代价。'},
 {t:'c',q:'NLG 步收益里，pre 是什么？',o:['做这道题之后的正确率','做这道题之前，这个知识点的平滑正确率','这道题的对错','全体学生的平均正确率'],a:1,why:'pre = (之前在这个知识点上答对的次数 + α × 总体正确率) / (之前做过的次数 + α)。post 是之后最多两次同知识点的平均正确率。'},
 {t:'c',q:'main() 为什么按“种子 0 跑完所有方法，再种子 1……”的顺序？',o:['更快','万一超时，每个方法至少都有一部分种子的结果，不会只跑完前几个方法','为了随机性','没有原因'],a:1,why:'代码注释写的是 breadth first（广度优先）。'},
 {t:'c',q:'load_state_dict(base_state, strict=False) 里为什么要 strict=False？',o:['为了更快','基础 DKT 没有策略头 g 的参数，严格模式会报 Missing key；放宽后 g 保持初始的 0','为了随机初始化','strict=False 会自动训练 g'],a:1,why:'第 11 天见过 Missing key(s) in state_dict 的报错。'},
 {t:'c',q:'paired_test 里的“配对”是指？',o:['两两比较所有模型','同一个种子、同一个数据集上，两个方法的结果配成一对，看差值','两个数据集配对','训练集和测试集配对'],a:1,why:'这样种子带来的随机波动在差值里抵消掉了，检验更灵敏。Holm 校正再处理多次检验的问题。'},
 {t:'code',q:'写代码：照论文的 step_rewards 公式，算一步的 NLG 收益。pre = 0.6，post = 0.9，结果存成 v（post ≥ pre 时 (post − pre)/(1 − pre)，否则 (post − pre)/pre，最后截到 [−1, 1]）。再算 pre = 0.6、post = 0.3 时的 v2。',start:'def nlg(pre, post):\n    # 在这里写公式\n    \n\nv = nlg(0.6, 0.9)\nv2 = nlg(0.6, 0.3)\nv, v2',check:env=>{const g=n=>{const x=env.vars[n];return x==null?null:MT.isT(x)?x.data[0]:x.v};const v=g('v'),v2=g('v2');if(v==null||Math.abs(v-0.75)>1e-6)return'v 应该是 (0.9 − 0.6)/(1 − 0.6) = 0.75';if(v2==null||Math.abs(v2+0.5)>1e-6)return'v2 应该是 (0.3 − 0.6)/0.6 = −0.5';return true},why:'v = (post - pre) / max(1 - pre, 1e-3) if post >= pre else (post - pre) / max(pre, 1e-3)，再 max(-1.0, min(1.0, v))。提高到 0.9 用掉了 75% 的“提升空间”，降到 0.3 掉了一半。'},
],
};
function quizState(day){return store.get('quiz'+day,{})}
function quizSave(day,s){store.set('quiz'+day,s)}
function quizRunShape(code){const env=MT.newEnv();MT.runCell('import torch\nfrom torch import nn',env);const r=MT.runCell(code,env);return r.ok&&r.value&&MT.isT(r.value)?{shape:r.value.shape,text:r.text}:{err:r.text}}
function quizRender(day){const box=document.querySelector(`.dayquiz[data-day="${day}"]`);if(!box)return;const Q=QUIZ[day]||[],st=quizState(day);
  box.innerHTML=Q.map((x,i)=>{const s=st[i];let body='';
    if(x.t==='c')body=`<div class="opts vert">${x.o.map((o,k)=>`<button class="btn sm${s!=null&&k===x.a?' right':''}${s===k&&k!==x.a?' wrong':''}" data-i="${i}" data-k="${k}">${esc(o)}</button>`).join('')}</div>`;
    else if(x.t==='shape')body=`<pre class="calc" style="margin:0">${esc(x.code)}</pre><div class="row"><label class="small muted" for="qz${day}_${i}">结果的形状</label><input type="text" id="qz${day}_${i}" placeholder="例如 (3, 4)" size="16" value="${s&&s.g?esc(s.g):''}"><button class="btn sm primary" data-i="${i}" data-act="shape">检查</button></div>`;
    else body=`<textarea class="qzcode" id="qz${day}_${i}" rows="${x.start.split('\n').length+1}" spellcheck="false" aria-label="在这里写代码">${esc(s&&s.code||x.start)}</textarea><div class="row"><button class="btn sm primary" data-i="${i}" data-act="code">运行并检查</button></div>`;
    const ok=s!=null&&(x.t==='c'?s===x.a:s.ok);
    const fb=s==null?'':x.t==='c'?`<div class="fb">${ok?'<span class="ok">对。</span>':`<span class="bad">不对，答案是“${esc(x.o[x.a])}”。</span>`}${esc(x.why)}</div>`:`<div class="fb">${ok?'<span class="ok">对。</span>':`<span class="bad">${esc(s.msg||'不对')}</span>`}${ok||s.tries>1?esc(x.why):''}${s.out?`<pre class="calc" style="margin-top:6px">${esc(s.out)}</pre>`:''}</div>`;
    return`<div class="q"><div><b>${i+1}.</b> ${esc(x.q)}</div>${body}${fb}</div>`}).join('');
  const done=Q.filter((x,i)=>st[i]!=null).length,right=Q.filter((x,i)=>st[i]!=null&&(x.t==='c'?st[i]===x.a:st[i].ok)).length;
  const sc=document.querySelector(`.quizscore[data-day="${day}"]`);if(sc)sc.innerHTML=`已答 ${done} / ${Q.length}，答对 ${right} 道。<button class="btn sm" data-qreset="${day}">清空重做</button>`;
  box.querySelectorAll('button[data-k]').forEach(b=>b.onclick=()=>{const s=quizState(day);s[b.dataset.i]=+b.dataset.k;quizSave(day,s);quizRender(day)});
  box.querySelectorAll('button[data-act="shape"]').forEach(b=>b.onclick=()=>{const i=+b.dataset.i,x=Q[i],g=document.getElementById(`qz${day}_${i}`).value,r=quizRunShape(x.code),s=quizState(day);const prev=s[i]||{};
    const guess=(g.match(/\d+/g)||[]).map(Number);const ok=!!r.shape&&guess.join()===r.shape.join();s[i]={ok,g,tries:(prev.tries||0)+1,msg:ok?'':r.shape?`实际是 ${shapeStr(r.shape)}。`:'代码运行出错。',out:r.text||r.err};quizSave(day,s);quizRender(day)});
  box.querySelectorAll('button[data-act="code"]').forEach(b=>b.onclick=()=>{const i=+b.dataset.i,x=Q[i],code=document.getElementById(`qz${day}_${i}`).value,env=MT.newEnv();MT.runCell('import torch\nfrom torch import nn',env);const r=MT.runCell(code,env);const s=quizState(day);const prev=s[i]||{};
    const res=r.ok?x.check(env):'代码报错了：'+r.text.split('\n').pop();s[i]={ok:res===true,code,tries:(prev.tries||0)+1,msg:res===true?'':res,out:r.text};quizSave(day,s);quizRender(day)})}
document.addEventListener('click',e=>{const b=e.target.closest('[data-qreset]');if(b){quizSave(b.dataset.qreset,{});quizRender(b.dataset.qreset)}});

for(const d of DAYS)quizRender(d);
