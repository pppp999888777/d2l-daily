
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
