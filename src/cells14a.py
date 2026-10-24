CELLS = [
"import torch\nfrom torch import nn\ndef setw(m):\n    for i, (n, p) in enumerate(m.named_parameters()):\n        with torch.no_grad():\n            p.copy_(torch.linspace(-0.5, 0.5, p.numel()).reshape(p.shape) * (1 + 0.1 * i))\n    return m\nx = torch.tensor([[[1.0, -2.0]], [[0.5, 1.0]], [[-1.0, 0.3]]])",
"lstm = setw(nn.LSTM(2, 3))\nout, (h, c) = lstm(x)\nout.sum().backward()\nlstm, out, h, c, lstm.weight_hh_l0.grad, lstm.bias_ih_l0.grad",
"gru = setw(nn.GRU(2, 3))\nout, h = gru(x)\n(out ** 2).sum().backward()\nout, h, gru.weight_ih_l0.grad, gru.bias_hh_l0.grad",
"l2 = setw(nn.LSTM(2, 3, num_layers=2, batch_first=True))\nxb = x.transpose(0, 1) if False else torch.tensor([[[1.0, -2.0], [0.5, 1.0], [-1.0, 0.3]]])\nout, (h, c) = l2(xb)\nout.sum().backward()\nl2, out.shape, h.shape, out, l2.weight_ih_l1.grad[:2]",
"emb = nn.Embedding(6, 3)\nwith torch.no_grad():\n    emb.weight.copy_(torch.arange(18.0).reshape(6, 3))\nidx = torch.tensor([[0, 5, 5], [2, 0, 1]])\ne = emb(idx)\n(e * torch.arange(3.0)).sum().backward()\nemb, e.shape, e[0], emb.weight.grad",
"r = setw(nn.RNN(2, 3, nonlinearity='relu'))\nr(x)[0][:, 0], r",
"isinstance(lstm, nn.LSTM), isinstance(gru, nn.LSTM), type(gru) == nn.GRU, lstm.hidden_size, lstm.num_layers",
"out, (h, c) = lstm(x, (torch.ones(1, 1, 3) * 0.2, torch.ones(1, 1, 3) * -0.3))\nh, c",
]
