'''
在TransH_pytorch的基础上，更进一步采用pytorch.nn.embedding class来实现transH模型

'''

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

import codecs
import numpy as np
import copy
import time
import random

entity2id = {}
relation2id = {}
relation_tph = {}
relation_hpt = {}

def data_loader(file):
    print("load file...")
    file1 = file + "train.txt"
    file2 = file + "entity2id.txt"
    file3 = file + "relation2id.txt"

    with open(file2, 'r') as f1, open(file3, 'r') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()
        for line in lines1:
            line = line.strip().split('\t')
            if len(line) != 2:
                continue
            entity2id[line[0]] = line[1]

        for line in lines2:
            line = line.strip().split('\t')
            if len(line) != 2:
                continue
            relation2id[line[0]] = line[1]

    entity_set = set()
    relation_set = set()
    triple_list = []
    relation_head = {}
    relation_tail = {}

    with codecs.open(file1, 'r') as f:
        content = f.readlines()
        for line in content:
            triple = line.strip().split("\t")
            if len(triple) != 3:
                continue

            h_ = int(entity2id[triple[0]])
            t_ = int(entity2id[triple[1]])
            r_ = int(relation2id[triple[2]])

            triple_list.append([h_, t_, r_])

            entity_set.add(h_)
            entity_set.add(t_)

            relation_set.add(r_)
            if r_ in relation_head:
                if h_ in relation_head[r_]:
                    relation_head[r_][h_] += 1
                else:
                    relation_head[r_][h_] = 1
            else:
                relation_head[r_] = {}
                relation_head[r_][h_] = 1

            if r_ in relation_tail:
                if t_ in relation_tail[r_]:
                    relation_tail[r_][t_] += 1
                else:
                    relation_tail[r_][t_] = 1
            else:
                relation_tail[r_] = {}
                relation_tail[r_][t_] = 1

    for r_ in relation_head:
        sum1, sum2 = 0, 0
        for head in relation_head[r_]:
            sum1 += 1
            sum2 += relation_head[r_][head]
        tph = sum2/sum1
        relation_tph[r_] = tph

    for r_ in relation_tail:
        sum1, sum2 = 0, 0
        for tail in relation_tail[r_]:
            sum1 += 1
            sum2 += relation_tail[r_][tail]
        hpt = sum2/sum1
        relation_hpt[r_] = hpt

    print("Complete load. entity : %d , relation : %d , triple : %d" % (
        len(entity_set), len(relation_set), len(triple_list)))

    if len(entity_set) != len(entity2id):
        raise ValueError("The number of entities is not equal")
    if len(relation_set) != len(relation2id):
        raise ValueError("The number of relations is not equal")

    return entity_set, relation_set, triple_list

def norm_l1(h, r, t):
    return np.sum(np.fabs(h + r - t))


class H(nn.Module):
    def __init__(self, entity_num, relation_num, dimension, margin, C):
        super(H, self).__init__()
        self.dimension = dimension
        self.margin = margin
        self.C = C


        self.relation_norm_embedding = torch.nn.Embedding(num_embeddings=relation_num,
                                                          embedding_dim=self.dimension).requires_grad_(True)
        self.relation_hyper_embedding = torch.nn.Embedding(num_embeddings=relation_num,
                                                           embedding_dim=self.dimension).requires_grad_(True)
        self.entity_embedding = torch.nn.Embedding(num_embeddings=entity_num,
                                                           embedding_dim=self.dimension).requires_grad_(True)
        self.loss_F = nn.MarginRankingLoss(self.margin, reduction="sum")
        self.distance_function = nn.PairwiseDistance(2)

    def normalization_norm_relations(self):
        norm = self.relation_norm_embedding.weight.detach().cpu().numpy()
        norm = norm / np.sqrt(np.sum(np.square(norm), axis=1, keepdims=True))
        self.relation_norm_embedding.weight.data.copy_(torch.from_numpy(norm))

    def distance(self, h, r_norm, r_hyper, t):
        # 在 tensor 的指定维度操作就是对指定维度包含的元素进行操作，如果想要保持结果的维度不变，设置参数keepdim=True即可
        # 如 下面sum中 r_norm * h 结果是一个1024 *50的矩阵（2维张量） sum在dim的结果就变成了 1024的向量（1位张量） 如果想和r_norm对应元素两两相乘
        # 就需要sum的结果也是2维张量 因此需要使用keepdim= True报纸张量的维度不变
        # 另外关于 dim 等于几表示最开始张量的第几个左括号，具体可以参考这个https://www.cnblogs.com/flix/p/11262606.html
        head = h - torch.sum(r_norm * h, dim=1, keepdim=True) * r_norm
        tail = t - torch.sum(r_norm * t, dim=1, keepdim=True) * r_norm
        return self.distance_function(head + r_hyper, tail)
        # return torch.sum(head + r_hyper - tail, dim=1, keepdim=True)

    def scalar(self, entity):
        return torch.sum(torch.relu(torch.norm(entity, p=2, dim=1, keepdim=False) - 1))

    def forward(self, current_triples, corrupted_triples):
        h, t, r = torch.chunk(current_triples, 3, dim=1)
        h_c, t_c, r_c = torch.chunk(corrupted_triples, 3, dim=1)

        # torch.nn.embedding类的forward只接受longTensor类型的张量
        head = torch.squeeze(self.entity_embedding(h), dim=1)
        r_norm = torch.squeeze(self.relation_norm_embedding(r), dim=1)
        r_hyper = torch.squeeze(self.relation_hyper_embedding(r), dim=1)
        tail = torch.squeeze(self.entity_embedding(t), dim=1)

        corrupted_head = torch.squeeze(self.entity_embedding(h_c), dim=1)
        corrupted_tail = torch.squeeze(self.entity_embedding(t_c), dim=1)

        pos = self.distance(head, r_norm, r_hyper, tail)
        neg = self.distance(corrupted_head, r_norm, r_hyper, corrupted_tail)

        # loss_F = max(0, -y*(x1-x2) + margin)
        loss1 = torch.sum(torch.relu(pos - neg + self.margin))
        loss = self.loss_F(neg, pos, torch.ones(1))\
                  + self.C * (self.scalar(head) + self.scalar(tail) + self.scalar(corrupted_head) + self.scalar(corrupted_tail))
                # + self.C * (torch.sum(F.relu(torch.norm(self.entity_embedding.weight, p=2, dim=1, keepdim=False) - 1)))
        print(loss1, loss)
        return torch.sum(loss)



class TransH:
    def __init__(self, entity_set, relation_set, triple_list, embedding_dim=50, lr=0.01, margin=1.0, norm=1, C=1.0, epsilon = 1e-5):
        self.entities = entity_set
        self.relations = relation_set
        self.triples = triple_list
        self.dimension = embedding_dim
        self.learning_rate = lr
        self.margin = margin
        self.norm = norm
        self.loss = 0.0
        self.entity_embedding = {}
        self.norm_relations = {}
        self.hyper_relations = {}
        self.C = C
        self.epsilon = epsilon


    def data_initialise(self):
        self.model = H(len(self.entities), len(self.relations), self.dimension, self.margin, self.C)
        self.optim = optim.SGD(self.model.parameters(), lr=self.learning_rate)


    def training_run(self, epochs=1, nbatches=100):

        batch_size = int(len(self.triples) / nbatches)
        print("batch size: ", batch_size)
        for epoch in range(epochs):
            start = time.time()
            self.loss = 0.0
            self.model.normalization_norm_relations()


            for batch in range(nbatches):
                batch_samples = random.sample(self.triples, batch_size)

                current = []
                corrupted = []
                change = False
                for sample in batch_samples:
                    corrupted_sample = copy.deepcopy(sample)
                    pr = np.random.random(1)[0]
                    p = relation_tph[int(corrupted_sample[2])] / (
                                relation_tph[int(corrupted_sample[2])] + relation_hpt[int(corrupted_sample[2])])
                    if pr > p:
                        # change the head entity
                        corrupted_sample[0] = random.sample(self.entities, 1)[0]
                        while corrupted_sample[0] == sample[0]:
                            corrupted_sample[0] = random.sample(self.entities, 1)[0]
                    else:
                        # change the tail entity
                        corrupted_sample[1] = random.sample(self.entities, 1)[0]
                        while corrupted_sample[1] == sample[1]:
                            corrupted_sample[1] = random.sample(self.entities, 1)[0]

                    current.append(sample)
                    corrupted.append(corrupted_sample)
                current = torch.from_numpy(np.array(current)).long()
                corrupted =  torch.from_numpy(np.array(corrupted)).long()
                self.update_triple_embedding(current, corrupted)
                end = time.time()
                print("epoch: ", epoch, "cost time: %s" % (round((end - start), 3)))
            print("running loss: ", self.loss)

        # .detach()的作用就是返回一个新的tensor，和原来tensor共享内存
        with codecs.open("TransH_pytorch_entity_" + str(self.dimension) + "dim_batch" + str(batch_size), "w") as f1:

            for i, e in enumerate(self.model.entity_embedding.weight):
                f1.write(str(i) + "\t")
                f1.write(str(e.detach().numpy().tolist()))
                f1.write("\n")

        with codecs.open("TransH_pytorch_norm_relations_" + str(self.dimension) + "dim_batch" + str(batch_size), "w") as f2:

            for i, e in enumerate(self.model.relation_norm_embedding.weight):
                f2.write(str(i) + "\t")
                f2.write(str(e.detach().numpy().tolist()))
                f2.write("\n")

        with codecs.open("TransH_pytorch_hyper_relations_" + str(self.dimension) + "dim_batch" + str(batch_size), "w") as f3:
            for i, e in enumerate(self.model.relation_hyper_embedding.weight):
                f3.write(str(i) + "\t")
                f3.write(str(e.detach().numpy().tolist()))
                f3.write("\n")

    def norm_l2(self, h, r_norm, r_hyper, t):
        return torch.norm(h - r_norm.dot(h)*r_norm + r_hyper -(t - r_norm.dot(t)*r_norm))


    # 知乎上询问过清华的大佬对于软约束项的建议 模长约束对结果收敛有影响，但是正交约束影响很小所以模长约束保留，正交约束可以不加
    def scale_entity(self, vector):
        return torch.relu(torch.sum(vector**2) - 1)

    def orthogonality(self, norm, hyper):
        return np.dot(norm, hyper)**2/np.linalg.norm(hyper)**2 - self.epsilon**2

    def update_triple_embedding(self, correct_sample, corrupted_sample):
        self.optim.zero_grad()
        loss = self.model(correct_sample, corrupted_sample)
        print(loss)
        self.loss += loss
        loss.backward()
        self.optim.step()



if __name__ == '__main__':
    file1 = "FB15k\\"
    entity_set, relation_set, triple_list = data_loader(file1)

    transH = TransH(entity_set, relation_set, triple_list, embedding_dim=50, lr=0.01, margin=1.0, norm=1)
    transH.data_initialise()
    transH.training_run()







# 关于叶节点的说明， 整个计算图中，只有叶节点的变量才能进行自动微分得到梯度，任何变量进行运算操作后，再把值付给他自己，这个变量就不是叶节点了，就不能进行自动微分







# print(torch.__version__)
#
# x = torch.randn(5000, 50, requires_grad=True)
# x_c = torch.randn(5000, 50, requires_grad=True)
# y = torch.randn(5000, 50, requires_grad=True)
# dr = torch.randn(5000, 50, requires_grad=True)
# nr = torch.randn(5000, 50, requires_grad=True)
#
# opt1 = optim.SGD([x], lr=0.01)
# opt2 = optim.SGD([y], lr=0.01)
# opt3 = optim.SGD([dr], lr=0.01)
# opt4 = optim.SGD([nr], lr=0.01)
# opt5 = optim.SGD([x_c], lr=0.01)
# for epoch in range(10):
#     for i in range(5000):
#         opt1.zero_grad()
#         opt2.zero_grad()
#         opt3.zero_grad()
#         opt4.zero_grad()
#         opt5.zero_grad()
#         # h =  torch.sum(torch.square(x - nr.dot(x) * nr + dr - (y - nr.dot(y) * nr)))
#         out = torch.sum(F.relu_(1 +  torch.norm(x-nr.dot(x[i])*nr[i] + dr[i] - (y -  nr.dot(y)*nr)) - torch.norm(x_c-nr.dot(x_c)*nr + dr - (y -  nr.dot(y)*nr))))
#         out.backward()
#         opt1.step()
#         opt2.step()
#         opt3.step()
#         opt4.step()
#         opt5.step()







