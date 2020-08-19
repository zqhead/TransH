import torch
import numpy as np
from torch.autograd import variable
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

print(torch.__version__)

x = torch.randn(50)
x_c = torch.randn(50, requires_grad=True)
y = torch.randn(50, requires_grad=True)
dr = torch.randn(50, requires_grad=True)
nr = torch.randn(50, requires_grad=True)

# opt1 = optim.SGD([x], lr=0.01)
# opt2 = optim.SGD([y], lr=0.01)
# opt3 = optim.SGD([dr], lr=0.01)
# opt4 = optim.SGD([nr], lr=0.01)
# opt5 = optim.SGD([x_c], lr=0.01)
# for epoch in range(10):
#
#     # h =  torch.sum(torch.square(x - nr.dot(x) * nr + dr - (y - nr.dot(y) * nr)))
# out =  torch.norm(x-nr.dot(x)*nr + dr - (y -  nr.dot(y)*nr))
# z = np.sum(np.square(x.detach().numpy() - np.dot(nr.detach().numpy(), x.detach().numpy()) * nr.detach().numpy()  + dr.detach().numpy() - y.detach().numpy() + np.dot(nr.detach().numpy(), y.detach().numpy()) * nr.detach().numpy()))
# print(out, z)



# def n(vector):
#     hyperWeight = vector.weight.detach().cpu().numpy()
#     hyperWeight = hyperWeight / np.sqrt(np.sum(np.square(hyperWeight), axis=1, keepdims=True))
#     relationHyper.weight.data.copy_(torch.from_numpy(hyperWeight))

class transH(nn.Module):
    def __init__(self ):
        super(transH, self).__init__()

        self.relationHyper = torch.nn.Embedding(num_embeddings=6, embedding_dim=10).requires_grad_(True)
        self.f = torch.nn.MarginRankingLoss(margin=1.00, reduction="sum")

    def forward(self, x):
        h, y, t = torch.chunk(x, 3, dim=1)
        a = torch.squeeze(self.relationHyper(h), dim=0)
        b = torch.squeeze(self.relationHyper(y), dim=0)
        c = torch.squeeze(self.relationHyper(t), dim=0)

        x1 = a + b
        x2 = c
        result = self.f(x1, x2, torch.ones(1))
        return result

model = transH()
opt1 = optim.SGD(model.parameters(), lr=0.01)
# n(relationHyper)
T = [1,2,3]
weight = torch.LongTensor([[0, 1, 2]])
weight1 = torch.LongTensor([0, 1, 2])
print(T[weight1[0]])
for epoch in range(100):
    opt1.zero_grad()
    z = model(weight)
    z.backward()
    opt1.step()
    print(z)
    # print(model.relationHyper.weight[0], model.relationHyper.weight[3])


#
# print(a)
# print(relationHyper.weight[0].shape, a.shape)