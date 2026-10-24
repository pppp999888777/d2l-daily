CELLS = [
"import torch\nimport torch.nn.functional as F\nfrom torch import nn\n\n\nclass CenteredLayer(nn.Module):\n    def __init__(self):\n        super().__init__()\n\n    def forward(self, X):\n        return X - X.mean()",
"layer = CenteredLayer()\nlayer(torch.FloatTensor([1, 2, 3, 4, 5]))",
"net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())\nY = net(torch.rand(4, 8))\nabs(Y.mean().item()) < 1e-6, net",
"class MyLinear(nn.Module):\n    def __init__(self, in_units, units):\n        super().__init__()\n        self.weight = nn.Parameter(torch.randn(in_units, units))\n        self.bias = nn.Parameter(torch.randn(units,))\n    def forward(self, X):\n        linear = torch.matmul(X, self.weight.data) + self.bias.data\n        return F.relu(linear)",
"linear = MyLinear(5, 3)\nlinear.weight.shape, linear.bias.requires_grad, [n for n, _ in linear.named_parameters()]",
"nn.init.constant_(linear.weight, 0.5); nn.init.zeros_(linear.bias)\nlinear(torch.ones(2, 5))",
"net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))\nnet(torch.rand(2, 64)).shape, net",
"class MLP(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.hidden = nn.Linear(20, 256)\n        self.output = nn.Linear(256, 10)\n\n    def forward(self, x):\n        return self.output(F.relu(self.hidden(x)))\n\nnet = MLP()\nnn.init.constant_(net.output.bias, 0.5)\nbest_wrong = net.state_dict()          # 没有 clone\nbest_right = {k: v.detach().clone() for k, v in net.state_dict().items()}\nwith torch.no_grad():\n    net.output.bias.fill_(7.0)\nprint(best_wrong['output.bias'][:3])\nprint(best_right['output.bias'][:3])",
"net.load_state_dict(best_right)\nnet.output.bias[:3]",
"class KTBase(nn.Module):\n    def __init__(self, K):\n        super().__init__()\n        self.K = K\n\nclass Tiny(KTBase):\n    def __init__(self, K, d=4):\n        super().__init__(K)\n        self.emb = nn.Linear(2 * K, d)\n        self.out = nn.Linear(d, K)\n    def encode(self, x):\n        return torch.relu(self.emb(x))\n\nm = Tiny(3)\nm.K, m, [n for n, _ in m.named_parameters()]",
"m(torch.ones(1, 6))",
]
