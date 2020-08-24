import json
import operator # operator模块输出一系列对应Python内部操作符的函数
import time

import numpy as np
import codecs

from TransH import data_loader,entity2id,relation2id


def test_data_loader(entity_embedding_file, norm_relation_embedding_file, hyper_relation_embedding_file, test_data_file):
    print("load data...")
    file1 = entity_embedding_file
    file2 = norm_relation_embedding_file
    file3 = hyper_relation_embedding_file
    file4 = test_data_file

    entity_dic = {}
    norm_relation = {}
    hyper_relation = {}
    triple_list = []

    with codecs.open(file1, 'r') as f1, codecs.open(file2, 'r') as f2, codecs.open(file3, 'r') as f3:
        lines1 = f1.readlines()
        lines2 = f2.readlines()
        lines3 = f3.readlines()
        for line in lines1:
            line = line.strip().split('\t')
            if len(line) != 2:
                continue
            entity_dic[line[0]] = json.loads(line[1])

        for line in lines2:
            line = line.strip().split('\t')
            if len(line) != 2:
                continue
            norm_relation[line[0]] = json.loads(line[1])

        for line in lines3:
            line = line.strip().split('\t')
            if len(line) != 2:
                continue
            hyper_relation[line[0]] = json.loads(line[1])

    with codecs.open(file4, 'r') as f4:
        content = f4.readlines()
        for line in content:
            triple = line.strip().split("\t")
            if len(triple) != 3:
                continue

            head = entity2id[triple[0]]
            tail = entity2id[triple[1]]
            relation = relation2id[triple[2]]

            triple_list.append([head, tail, relation])

    print("Complete load. entity : %d , relation : %d , triple : %d" % (
        len(entity_dic), len(norm_relation), len(triple_list)))

    return entity_dic, norm_relation, hyper_relation, triple_list

class testTransH:
    def __init__(self, entities_dict, norm_relation, hyper_relation, test_triple_list, train_triple_list, filter_triple=False, n=2500 ,norm=1):
        self.entities = entities_dict
        self.norm_relation = norm_relation
        self.hyper_relation = hyper_relation
        self.test_triples = test_triple_list
        self.train_triples = train_triple_list
        self.filter = filter_triple
        self.norm = norm
        self.n = n
        self.mean_rank = 0
        self.hit_10 = 0

    def test_theading(self, test_triple):
        hits = 0
        rank_sum = 0
        num = 0

        for triple in test_triple:
            num += 1
            print(num, triple)
            rank_head_dict = {}
            rank_tail_dict = {}
            #
            for entity in self.entities.keys():
                head_triple = [entity, triple[1], triple[2]]
                if self.filter:
                    if head_triple in self.train_triples:
                        continue
                head_embedding = self.entities[head_triple[0]]
                tail_embedding = self.entities[head_triple[1]]
                norm_relation = self.norm_relation[head_triple[2]]
                hyper_relation = self.hyper_relation[head_triple[2]]
                distance = self.distance(head_embedding, norm_relation,hyper_relation, tail_embedding)
                rank_head_dict[tuple(head_triple)] = distance

            for tail in self.entities.keys():
                tail_triple = [triple[0], tail, triple[2]]
                if self.filter:
                    if tail_triple in self.train_triples:
                        continue
                head_embedding = self.entities[tail_triple[0]]
                tail_embedding = self.entities[tail_triple[1]]
                norm_relation = self.norm_relation[tail_triple[2]]
                hyper_relation = self.hyper_relation[tail_triple[2]]
                distance = self.distance(head_embedding, norm_relation, hyper_relation, tail_embedding)
                rank_tail_dict[tuple(tail_triple)] = distance

            # itemgetter 返回一个可调用对象，该对象可以使用操作__getitem__()方法从自身的操作中捕获item
            # 使用itemgetter()从元组记录中取回特定的字段 搭配sorted可以将dictionary根据value进行排序
            # sort 是应用在 list 上的方法，sorted 可以对所有可迭代的对象进行排序操作。
            '''
            sorted(iterable, cmp=None, key=None, reverse=False)
            参数说明：
            iterable -- 可迭代对象。
            cmp -- 比较的函数，这个具有两个参数，参数的值都是从可迭代对象中取出，此函数必须遵守的规则为，大于则返回1，小于则返回-1，等于则返回0。
            key -- 主要是用来进行比较的元素，只有一个参数，具体的函数的参数就是取自于可迭代对象中，指定可迭代对象中的一个元素来进行排序。
            reverse -- 排序规则，reverse = True 降序 ， reverse = False 升序（默认）。
            '''
            rank_head_sorted = sorted(rank_head_dict.items(), key=operator.itemgetter(1), reverse=False)
            rank_tail_sorted = sorted(rank_tail_dict.items(), key=operator.itemgetter(1), reverse=False)

            # calculate the mean_rank and hit_10
            # head data
            for i in range(len(rank_head_sorted)):
                if triple[0] == rank_head_sorted[i][0][0]:
                    if i < 10:
                        hits += 1
                    rank_sum = rank_sum + i + 1
                    break

            # tail rank
            for i in range(len(rank_tail_sorted)):
                if triple[1] == rank_tail_sorted[i][0][1]:
                    if i < 10:
                        hits += 1
                    rank_sum = rank_sum + i + 1
                    break
        return hits, rank_sum


    def test_run(self):
        hits = 0
        rank_sum = 0
        num = 0

        for triple in self.test_triples:
            start = time.time()
            num += 1
            print(num, triple)
            rank_head_dict = {}
            rank_tail_dict = {}
            #
            for entity in self.entities.keys():

                head_triple = [entity, triple[1], triple[2]]
                if self.filter:
                    if head_triple in self.train_triples:
                        continue
                head_embedding = self.entities[head_triple[0]]
                tail_embedding = self.entities[head_triple[1]]
                norm_relation = self.norm_relation[head_triple[2]]
                hyper_relation = self.hyper_relation[head_triple[2]]
                distance = self.distance(head_embedding, norm_relation,hyper_relation, tail_embedding)
                rank_head_dict[tuple(head_triple)] = distance


            for tail in self.entities.keys():
                tail_triple = [triple[0], tail, triple[2]]
                if self.filter:
                    if tail_triple in self.train_triples:
                        continue
                head_embedding = self.entities[tail_triple[0]]
                tail_embedding = self.entities[tail_triple[1]]
                norm_relation = self.norm_relation[tail_triple[2]]
                hyper_relation = self.hyper_relation[tail_triple[2]]
                distance = self.distance(head_embedding, norm_relation, hyper_relation, tail_embedding)
                rank_tail_dict[tuple(tail_triple)] = distance

            # itemgetter 返回一个可调用对象，该对象可以使用操作__getitem__()方法从自身的操作中捕获item
            # 使用itemgetter()从元组记录中取回特定的字段 搭配sorted可以将dictionary根据value进行排序
            # sort 是应用在 list 上的方法，sorted 可以对所有可迭代的对象进行排序操作。
            '''
            sorted(iterable, cmp=None, key=None, reverse=False)
            参数说明：
            iterable -- 可迭代对象。
            cmp -- 比较的函数，这个具有两个参数，参数的值都是从可迭代对象中取出，此函数必须遵守的规则为，大于则返回1，小于则返回-1，等于则返回0。
            key -- 主要是用来进行比较的元素，只有一个参数，具体的函数的参数就是取自于可迭代对象中，指定可迭代对象中的一个元素来进行排序。
            reverse -- 排序规则，reverse = True 降序 ， reverse = False 升序（默认）。
            '''
            rank_head_sorted = sorted(rank_head_dict.items(), key=operator.itemgetter(1), reverse=False)
            rank_tail_sorted = sorted(rank_tail_dict.items(), key=operator.itemgetter(1), reverse=False)

            # calculate the mean_rank and hit_10
            # head data
            for i in range(len(rank_head_sorted)):
                if triple[0] == rank_head_sorted[i][0][0]:
                    if i < 10:
                        hits += 1
                    rank_sum = rank_sum + i + 1
                    break

            # tail rank
            for i in range(len(rank_tail_sorted)):
                if triple[1] == rank_tail_sorted[i][0][1]:
                    if i < 10:
                        hits += 1
                    rank_sum = rank_sum + i + 1
                    break
            end = time.time()
            print("epoch: ", num, "cost time: %s" % (round((end - start), 3)))
        self.hit_10 = hits / (2 * len(self.test_triples))
        self.mean_rank = rank_sum / (2 * len(self.test_triples))

        return self.hit_10, self.mean_rank


    def distance(self, h, r_norm, r_hyper, t):
        head = np.array(h)
        norm = np.array(r_norm)
        hyper = np.array(r_hyper)
        tail = np.array(t)
        h_hyper = head - np.dot(norm, head) * norm
        t_hyper = tail - np.dot(norm, tail) * norm
        d = h_hyper + hyper - t_hyper
        return np.sum(np.square(d))



if __name__ == "__main__":
    _, _, train_triple = data_loader("FB15k\\")

    entity, norm_relation, hyper_relation, test_triple = test_data_loader("TransH_pytorch_entity_50dim_batch4831",
                                                               "TransH_pytorch_norm_relations_50dim_batch4831",
                                                               "TransH_pytorch_hyper_relations_50dim_batch4831",
                                                               "FB15k\\test.txt")

    test = testTransH(entity, norm_relation, hyper_relation, test_triple, train_triple, filter_triple=False, n=2500, norm=2)
    hit10, mean_rank = test.test_run()
    print("raw entity hits@10: ", hit10)
    print("raw entity meanrank: ",mean_rank)

    # test2 = testTransH(entity, norm_relation, hyper_relation, test_triple, train_triple, filter_triple=True, n=2500, norm=2)
    # filter_hit10, filter_mean_rank = test2.test_run()
    # print("filter entity hits@10: ", filter_hit10)
    # print("filter entity meanrank: ", filter_mean_rank)


# import torch
# import numpy as np
# from torch.autograd import variable
# import torch.optim as optim
# import torch.nn.functional as F
# import torch.nn as nn
#
# print(torch.__version__)
#
# x = torch.randn(50)
# x_c = torch.randn(50, requires_grad=True)
# y = torch.randn(50, requires_grad=True)
# dr = torch.randn(50, requires_grad=True)
# nr = torch.randn(50, requires_grad=True)
#
# # opt1 = optim.SGD([x], lr=0.01)
# # opt2 = optim.SGD([y], lr=0.01)
# # opt3 = optim.SGD([dr], lr=0.01)
# # opt4 = optim.SGD([nr], lr=0.01)
# # opt5 = optim.SGD([x_c], lr=0.01)
# # for epoch in range(10):
# #
# #     # h =  torch.sum(torch.square(x - nr.dot(x) * nr + dr - (y - nr.dot(y) * nr)))
# # out =  torch.norm(x-nr.dot(x)*nr + dr - (y -  nr.dot(y)*nr))
# # z = np.sum(np.square(x.detach().numpy() - np.dot(nr.detach().numpy(), x.detach().numpy()) * nr.detach().numpy()  + dr.detach().numpy() - y.detach().numpy() + np.dot(nr.detach().numpy(), y.detach().numpy()) * nr.detach().numpy()))
# # print(out, z)
#
#
#
# # def n(vector):
# #     hyperWeight = vector.weight.detach().cpu().numpy()
# #     hyperWeight = hyperWeight / np.sqrt(np.sum(np.square(hyperWeight), axis=1, keepdims=True))
# #     relationHyper.weight.data.copy_(torch.from_numpy(hyperWeight))
#
# class transH(nn.Module):
#     def __init__(self ):
#         super(transH, self).__init__()
#
#         self.relationHyper = torch.nn.Embedding(num_embeddings=6, embedding_dim=10).requires_grad_(True)
#         self.f = torch.nn.MarginRankingLoss(margin=1.00, reduction="sum")
#
#     def forward(self, x):
#         h, y, t = torch.chunk(x, 3, dim=1)
#         a = torch.squeeze(self.relationHyper(h), dim=0)
#         b = torch.squeeze(self.relationHyper(y), dim=0)
#         c = torch.squeeze(self.relationHyper(t), dim=0)
#
#         x1 = a + b
#         x2 = c
#         result = self.f(x1, x2, torch.ones(1))
#         return result
#
# model = transH()
# opt1 = optim.SGD(model.parameters(), lr=0.01)
# # n(relationHyper)
# T = [1,2,3]
# weight = torch.LongTensor([[0, 1, 2]])
# weight1 = torch.LongTensor([0, 1, 2])
# print(T[weight1[0]])
# for epoch in range(100):
#     opt1.zero_grad()
#     z = model(weight)
#     z.backward()
#     opt1.step()
#     print(z)
#     # print(model.relationHyper.weight[0], model.relationHyper.weight[3])
#
#
# #
# # print(a)
# # print(relationHyper.weight[0].shape, a.shape)