'''
在TransH_pytorch的基础上，更进一步采用pytorch.nn.embedding class来实现transH模型

'''

import codecs
import numpy as np
import copy
import time
import random
import matplotlib.pyplot as plt
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import operator # operator模块输出一系列对应Python内部操作符的函数

entities2id = {}
relations2id = {}
relation_tph = {}
relation_hpt = {}


def dataloader(file1, file2, file3, file4):
    print("load file...")

    entity = []
    relation = []
    with open(file2, 'r') as f1, open(file3, 'r') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()
        for line in lines1:
            line = line.strip().split('\t')
            if len(line) != 2:
                continue
            entities2id[line[0]] = line[1]
            entity.append(int(line[1]))

        for line in lines2:
            line = line.strip().split('\t')
            if len(line) != 2:
                continue
            relations2id[line[0]] = line[1]
            relation.append(int(line[1]))


    triple_list = []
    relation_head = {}
    relation_tail = {}

    with codecs.open(file1, 'r') as f:
        content = f.readlines()
        for line in content:
            triple = line.strip().split("\t")
            if len(triple) != 3:
                continue

            h_ = int(entities2id[triple[0]])
            r_ = int(relations2id[triple[1]])
            t_ = int(entities2id[triple[2]])


            triple_list.append([h_, r_, t_])
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
        tph = sum2 / sum1
        relation_tph[r_] = tph

    for r_ in relation_tail:
        sum1, sum2 = 0, 0
        for tail in relation_tail[r_]:
            sum1 += 1
            sum2 += relation_tail[r_][tail]
        hpt = sum2 / sum1
        relation_hpt[r_] = hpt

    valid_triple_list = []
    with codecs.open(file4, 'r') as f:
        content = f.readlines()
        for line in content:
            triple = line.strip().split("\t")
            if len(triple) != 3:
                continue

            h_ = int(entities2id[triple[0]])
            r_ = int(relations2id[triple[1]])
            t_ = int(entities2id[triple[2]])


            valid_triple_list.append([h_, r_, t_])

    print("Complete load. entity : %d , relation : %d , train triple : %d, , valid triple : %d" % (
    len(entity), len(relation), len(triple_list), len(valid_triple_list)))

    return entity, relation, triple_list, valid_triple_list


class H(nn.Module):
    def __init__(self, entity_num, relation_num, dimension, margin, C, epsilon, norm):
        super(H, self).__init__()
        self.dimension = dimension
        self.margin = margin
        self.C = C
        self.entity_num = entity_num
        self.relation_num = relation_num
        self.epsilon = epsilon
        self.norm = norm

        self.relation_norm_embedding = torch.nn.Embedding(num_embeddings=relation_num,
                                                          embedding_dim=self.dimension).cuda()
        self.relation_hyper_embedding = torch.nn.Embedding(num_embeddings=relation_num,
                                                           embedding_dim=self.dimension).cuda()
        self.entity_embedding = torch.nn.Embedding(num_embeddings=entity_num,
                                                           embedding_dim=self.dimension).cuda()
        self.loss_F = nn.MarginRankingLoss(self.margin, reduction="mean").cuda()
        # pairwiseDIstance 用于计算成批成对的两个向量之间的距离（差值），具体的距离为 Lp范数，参数P定义了使用第几范数，默认为L2


        self.__data_init()

    def __data_init(self):
        nn.init.xavier_uniform_(self.relation_norm_embedding.weight.data)
        nn.init.xavier_uniform_(self.relation_hyper_embedding.weight.data)
        nn.init.xavier_uniform_(self.entity_embedding.weight.data)

    def input_pre_transe(self, ent_vector, rel_vector, rel_norm):
        for i in range(self.entity_num):
            self.entity_embedding.weight.data[i] = torch.from_numpy(np.array(ent_vector[i]))
        for i in range(self.relation_num):
            self.relation_hyper_embedding.weight.data[i] = torch.from_numpy(np.array(rel_vector[i]))
        for i in range(self.relation_num):
            self.relation_norm_embedding.weight.data[i] = torch.from_numpy(np.array(rel_norm[i]))

    # def normalization_norm_relations(self):
    #     norm = self.relation_norm_embedding.weight.detach().cpu().numpy()
    #     norm = norm / np.sqrt(np.sum(np.square(norm), axis=1, keepdims=True))
    #     self.relation_norm_embedding.weight.data.copy_(torch.from_numpy(norm))

    def projected(self, ent, norm):

        norm = F.normalize(norm, p=2, dim=-1)

        return ent - torch.sum(ent * norm, dim = 1, keepdim=True) * norm

    def distance(self, h, r, t):
        # 在 tensor 的指定维度操作就是对指定维度包含的元素进行操作，如果想要保持结果的维度不变，设置参数keepdim=True即可
        # 如 下面sum中 r_norm * h 结果是一个1024 *50的矩阵（2维张量） sum在dim的结果就变成了 1024的向量（1位张量） 如果想和r_norm对应元素两两相乘
        # 就需要sum的结果也是2维张量 因此需要使用keepdim= True报纸张量的维度不变
        # 另外关于 dim 等于几表示张量的第几个维度，从0开始计数，可以理解为张量的最开始的第几个左括号，具体可以参考这个https://www.cnblogs.com/flix/p/11262606.html
        head = self.entity_embedding(h)
        r_norm = self.relation_norm_embedding(r)
        r_hyper = self.relation_hyper_embedding(r)
        tail = self.entity_embedding(t)

        head_hyper = self.projected(head, r_norm)
        tail_hyper = self.projected(tail, r_norm)

        distance = head_hyper + r_hyper - tail_hyper
        score = torch.norm(distance, p = self.norm, dim=1)
        return score

    def test_distance(self, h, r, t):

        head = self.entity_embedding(h.cuda())
        r_norm = self.relation_norm_embedding(r.cuda())
        r_hyper = self.relation_hyper_embedding(r.cuda())
        tail = self.entity_embedding(t.cuda())

        head_hyper = self.projected(head, r_norm)
        tail_hyper = self.projected(tail, r_norm)

        distance = head_hyper + r_hyper - tail_hyper
        score = torch.norm(distance, p = self.norm, dim=1)
        return score.cpu().detach().numpy()

    def scale_loss(self, embedding):
        return torch.sum(
            torch.max(
                torch.sum(
                    embedding ** 2, dim=1, keepdim=True
                )-torch.autograd.Variable(torch.FloatTensor([1.0]).cuda()),
                torch.autograd.Variable(torch.FloatTensor([0.0]).cuda())
            ))


    def orthogonal_loss(self, relation_embedding, w_embedding):
        dot = torch.sum(relation_embedding * w_embedding, dim=1, keepdim=False) ** 2
        norm = torch.norm(relation_embedding, p=self.norm, dim=1) ** 2

        loss = torch.sum(torch.relu(dot / norm - torch.autograd.Variable(torch.FloatTensor([self.epsilon]).cuda() ** 2)))
        return loss
        # return torch.sum(
        #     torch.sum(
        #         relation_embedding * w_embedding, dim=1, keepdim=True
        #     ) ** 2 / torch.sum(relation_embedding ** 2, dim=1, keepdim=True)
        # )


    def forward(self, current_triples, corrupted_triples):
        h, r, t = torch.chunk(current_triples, 3, dim=1)
        h_c, r_c, t_c = torch.chunk(corrupted_triples, 3, dim=1)

        # torch.nn.embedding类的forward只接受longTensor类型的张量

        h = torch.squeeze(h, dim=1).cuda()
        r = torch.squeeze(r, dim=1).cuda()
        t = torch.squeeze(t, dim=1).cuda()
        h_c = torch.squeeze(h_c, dim=1).cuda()
        r_c = torch.squeeze(r_c, dim=1).cuda()
        t_c = torch.squeeze(t_c, dim=1).cuda()

        pos = self.distance(h, r, t)
        neg = self.distance(h_c, r_c, t_c)

        y = Variable(torch.Tensor([-1])).cuda()
        # loss_F = max(0, -y*(x1-x2) + margin)
        loss = self.loss_F(pos, neg, y)

        entity_embedding  = self.entity_embedding(torch.cat([h, t, h_c, t_c]).cuda())
        relation_embedding = self.relation_hyper_embedding(torch.cat([r, r_c]).cuda())
        w_embedding = self.relation_norm_embedding(torch.cat([r, r_c]).cuda())

        # orthogonal_loss = torch.sum(torch.relu(torch.sum(self.relation_norm_embedding.weight * self.relation_hyper_embedding.weight, dim=1, keepdim=False) ** 2
        #                / torch.norm(self.relation_hyper_embedding.weight, p=self.norm, dim=1) ** 2 - (torch.autograd.Variable(torch.FloatTensor([self.epsilon]).cuda()) ** 2)))
        #
        # scale_loss = torch.sum(torch.relu(torch.norm(self.entity_embedding.weight, p=self.norm, dim=1, keepdim=False) ** 2 - torch.autograd.Variable(torch.FloatTensor([1.0]).cuda())))

        orthogonal_loss = self.orthogonal_loss(relation_embedding, w_embedding)

        scale_loss = self.scale_loss(entity_embedding)


        # # 知乎上询问过清华的大佬对于软约束项的建议 模长约束对结果收敛有影响，但是正交约束影响很小.所以模长约束保留，正交约束可以不加
        return loss + self.C * (scale_loss/len(entity_embedding) + orthogonal_loss/len(relation_embedding))

class TransH:
    def __init__(self, entity_set, relation_set, triple_list, embedding_dim=50, lr=0.01, margin=1.0, norm=1, C=1.0, epsilon = 1e-5, valid_triple_list = None):
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
        self.valid_triples = valid_triple_list
        self.valid_loss = 0

        self.test_triples = []
        self.train_loss = []
        self.validation_loss = []


    def data_initialise(self):
        self.model = H(len(self.entities), len(self.relations), self.dimension, self.margin, self.C, self.epsilon, self.norm)
        self.optim = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # self.optim = optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def insert_data(self, file1, file2, file3, file4, file5):

        entity_dic = {}
        norm_relation = {}
        hyper_relation = {}
        with codecs.open(file1, 'r') as f1, codecs.open(file2, 'r') as f2, codecs.open(file3, 'r') as f3:
          lines1 = f1.readlines()
          lines2 = f2.readlines()
          lines3 = f3.readlines()
          for line in lines1:
              line = line.strip().split('\t')
              if len(line) != 2:
                  continue
              entity_dic[int(line[0])] = json.loads(line[1])

          for line in lines2:
              line = line.strip().split('\t')
              if len(line) != 2:
                  continue
              norm_relation[int(line[0])] = json.loads(line[1])

          for line in lines3:
              line = line.strip().split('\t')
              if len(line) != 2:
                  continue
              hyper_relation[int(line[0])] = json.loads(line[1])
        self.model.input_pre_transe(entity_dic, hyper_relation, norm_relation)

        triple_list = []
        with codecs.open(file4, 'r') as f4:
          content = f4.readlines()
          for line in content:
              triple = line.strip().split("\t")
              if len(triple) != 3:
                  continue

              head = int(entities2id[triple[0]])
              relation = int(relations2id[triple[1]])
              tail = int(entities2id[triple[2]])


              triple_list.append([head, relation, tail])

        self.test_triples = triple_list

        with codecs.open(file5, 'r') as f5:
            lines = f5.readlines()
            for line in lines:
                line = line.strip().split('\t')
                self.train_loss = json.loads(line[0])
                self.validation_loss = json.loads(line[1])
        print(self.train_loss, self.validation_loss)



    def training_run(self, epochs=300, batch_size=100, out_file_title = ''):

        n_batches = int(len(self.triples) / batch_size)
        valid_batch = int(len(self.valid_triples) / batch_size) + 1
        print("batch size: ", n_batches, "valid_batch: " , valid_batch)
        for epoch in range(epochs):
            start = time.time()
            self.loss = 0.0
            self.valid_loss = 0.0


            for batch in range(n_batches):
                batch_samples = random.sample(self.triples, batch_size)

                current = []
                corrupted = []
                for sample in batch_samples:
                    corrupted_sample = copy.deepcopy(sample)
                    pr = np.random.random(1)[0]
                    p = relation_tph[int(corrupted_sample[1])] / (
                                relation_tph[int(corrupted_sample[1])] + relation_hpt[int(corrupted_sample[1])])
                    if pr > p:
                        # change the head entity
                        corrupted_sample[0] = random.sample(self.entities, 1)[0]
                        while corrupted_sample[0] == sample[0]:
                            corrupted_sample[0] = random.sample(self.entities, 1)[0]
                    else:
                        # change the tail entity
                        corrupted_sample[2] = random.sample(self.entities, 1)[0]
                        while corrupted_sample[2] == sample[2]:
                            corrupted_sample[2] = random.sample(self.entities, 1)[0]

                    current.append(sample)
                    corrupted.append(corrupted_sample)

                current = torch.from_numpy(np.array(current)).long()
                corrupted =  torch.from_numpy(np.array(corrupted)).long()
                self.update_triple_embedding(current, corrupted)

            for batch in range(valid_batch):

                batch_samples = random.sample(self.valid_triples, batch_size)

                current = []
                corrupted = []
                for sample in batch_samples:
                    corrupted_sample = copy.deepcopy(sample)
                    pr = np.random.random(1)[0]
                    p = relation_tph[int(corrupted_sample[1])] / (
                            relation_tph[int(corrupted_sample[1])] + relation_hpt[int(corrupted_sample[1])])

                    '''
                    这里关于p的说明 tph 表示每一个头实体对应的平均尾实体数 hpt 表示每一个尾实体对应的平均头实体数
                    当tph > hpt 时 更倾向于替换头 反之则跟倾向于替换尾实体
                    
                    举例说明 
                    在一个知识图谱中，一共有10个实体 和n个关系，如果其中一个关系使两个头实体对应五个尾实体，
                    那么这些头实体的平均 tph为2.5，而这些尾实体的平均 hpt只有0.4，
                    则此时我们更倾向于替换头实体，
                    因为替换头实体才会有更高概率获得正假三元组，如果替换头实体，获得正假三元组的概率为 8/9 而替换尾实体获得正假三元组的概率只有 5/9
                    '''
                    if pr < p:
                        # change the head entity
                        corrupted_sample[0] = random.sample(self.entities, 1)[0]
                        while corrupted_sample[0] == sample[0]:
                            corrupted_sample[0] = random.sample(self.entities, 1)[0]
                    else:
                        # change the tail entity
                        corrupted_sample[2] = random.sample(self.entities, 1)[0]
                        while corrupted_sample[2] == sample[2]:
                            corrupted_sample[2] = random.sample(self.entities, 1)[0]

                    current.append(sample)
                    corrupted.append(corrupted_sample)

                current = torch.from_numpy(np.array(current)).long()
                corrupted = torch.from_numpy(np.array(corrupted)).long()
                self.calculate_valid_loss(current, corrupted)

            end = time.time()
            mean_train_loss = self.loss / n_batches
            mean_valid_loss = self.valid_loss / valid_batch
            print("epoch: ", epoch, "cost time: %s" % (round((end - start), 3)))
            print("Train loss: ", mean_train_loss, "Valid loss: ", mean_valid_loss)

            self.train_loss.append(float(mean_train_loss))
            self.validation_loss.append(float(mean_valid_loss))

        # visualize the loss as the network trained
        fig = plt.figure(figsize=(6, 4))
        plt.plot(range(1, len(self.train_loss) + 1), self.train_loss, label='Train Loss')
        plt.plot(range(1, len(self.validation_loss) + 1), self.validation_loss, label='Validation Loss')

        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.xlim(0, len(self.train_loss) + 1)  # consistent scale
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.title(out_file_title + " Training loss")
        plt.show()

        fig.savefig(out_file_title+'loss_plot.png', bbox_inches='tight')

        # .detach()的作用就是返回一个新的tensor，和原来tensor共享内存，但是这个张量会从计算途中分离出来，并且requires_grad=false
        # 由于 能被grad的tensor不能直接使用.numpy(), 所以要是用。detach().numpy()
        with codecs.open(out_file_title + "TransH_pytorch_entity_" + str(self.dimension) + "dim_batch" + str(batch_size), "w") as f1:

            for i, e in enumerate(self.model.entity_embedding.weight):
                f1.write(str(i) + "\t")
                f1.write(str(e.cpu().detach().numpy().tolist()))
                f1.write("\n")

        with codecs.open(out_file_title + "TransH_pytorch_norm_relations_" + str(self.dimension) + "dim_batch" + str(batch_size), "w") as f2:

            for i, e in enumerate(self.model.relation_norm_embedding.weight):
                f2.write(str(i) + "\t")
                f2.write(str(e.cpu().detach().numpy().tolist()))
                f2.write("\n")

        with codecs.open(out_file_title + "TransH_pytorch_hyper_relations_" + str(self.dimension) + "dim_batch" + str(batch_size), "w") as f3:
            for i, e in enumerate(self.model.relation_hyper_embedding.weight):
                f3.write(str(i) + "\t")
                f3.write(str(e.cpu().detach().numpy().tolist()))
                f3.write("\n")

        with codecs.open("Fb15k_loss_record.txt", "w") as f1:
                f1.write(str(self.train_loss)+ "\t" + str(self.validation_loss))

    def update_triple_embedding(self, correct_sample, corrupted_sample):
        self.optim.zero_grad()
        loss = self.model(correct_sample, corrupted_sample)
        self.loss += loss
        loss.backward()
        self.optim.step()

    def calculate_valid_loss(self, correct_sample, corrupted_sample):
        loss = self.model(correct_sample, corrupted_sample)
        self.valid_loss += loss

    def test_run(self, filter = False):

        self.filter = filter
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
            head_embedding = []
            tail_embedding = []
            norm_relation = []
            hyper_relation = []
            tamp = []

            head_filter = []
            tail_filter = []
            if self.filter:

                for tr in self.triples:
                    if tr[1] == triple[1] and tr[2] == triple[2] and tr[0] != triple[0]:
                        head_filter.append(tr)
                    if tr[0] == triple[0] and tr[1] == triple[1] and tr[2] != triple[2]:
                        tail_filter.append(tr)
                for tr in self.test_triples:
                    if tr[1] == triple[1] and tr[2] == triple[2] and tr[0] != triple[0]:
                        head_filter.append(tr)
                    if tr[0] == triple[0] and tr[1] == triple[1] and tr[2] != triple[2]:
                        tail_filter.append(tr)
                for tr in self.valid_triples:
                    if tr[1] == triple[1] and tr[2] == triple[2] and tr[0] != triple[0]:
                        head_filter.append(tr)
                    if tr[0] == triple[0] and tr[1] == triple[1] and tr[2] != triple[2]:
                        tail_filter.append(tr)

            for i, entity in enumerate(self.entities):

                head_triple = [entity, triple[1], triple[2]]
                if self.filter:
                    if head_triple in head_filter:
                        continue
                head_embedding.append(head_triple[0])
                norm_relation.append(head_triple[1])
                tail_embedding.append(head_triple[2])

                tamp.append(tuple(head_triple))

            head_embedding = torch.from_numpy(np.array(head_embedding)).long()
            norm_relation = torch.from_numpy(np.array(norm_relation)).long()
            tail_embedding = torch.from_numpy(np.array(tail_embedding)).long()
            distance  = self.model.test_distance(head_embedding, norm_relation, tail_embedding)

            for i in range(len(tamp)):
                rank_head_dict[tamp[i]] = distance[i]

            head_embedding = []
            tail_embedding = []
            norm_relation = []
            hyper_relation = []
            tamp = []

            for i, tail in enumerate(self.entities):

                tail_triple = [triple[0], triple[1], tail]
                if self.filter:
                    if tail_triple in tail_filter:
                        continue
                head_embedding.append(tail_triple[0])
                norm_relation.append(tail_triple[1])
                tail_embedding.append(tail_triple[2])
                tamp.append(tuple(tail_triple))

            head_embedding = torch.from_numpy(np.array(head_embedding)).long()
            norm_relation = torch.from_numpy(np.array(norm_relation)).long()
            tail_embedding = torch.from_numpy(np.array(tail_embedding)).long()

            distance  = self.model.test_distance(head_embedding, norm_relation, tail_embedding)
            for i in range(len(tamp)):
                rank_tail_dict[tamp[i]] = distance[i]

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
            i = 0
            for i in range(len(rank_head_sorted)):
                if triple[0] == rank_head_sorted[i][0][0]:
                    if i < 10:
                        hits += 1
                    rank_sum = rank_sum + i + 1
                    break

            # tail rank
            i = 0
            for i in range(len(rank_tail_sorted)):
                if triple[2] == rank_tail_sorted[i][0][2]:
                    if i < 10:
                        hits += 1
                    rank_sum = rank_sum + i + 1
                    break
            end = time.time()
            print("epoch: ", num, "cost time: %s" % (round((end - start), 3)), str(hits / (2 * num)),
                  str(rank_sum / (2 * num)))
        self.hit_10 = hits / (2 * len(self.test_triples))
        self.mean_rank = rank_sum / (2 * len(self.test_triples))

        return self.hit_10, self.mean_rank



if __name__ == '__main__':
    # file1 = "WN18\\wordnet-mlj12-train.txt"
    # file2 = "WN18\\entity2id.txt"
    # file3 = "WN18\\relation2id.txt"
    # file4 = "WN18\\wordnet-mlj12-valid.txt"

    file1 = "FB15k\\freebase_mtr100_mte100-train.txt"
    file2 = "FB15k\\entity2id.txt"
    file3 = "FB15k\\relation2id.txt"
    file4 = "FB15k\\freebase_mtr100_mte100-valid.txt"
    entity_set, relation_set, triple_list, valid_triple_list = dataloader(file1, file2, file3, file4)

    # file5 = ""
    # file6 = ""
    # file7 = ""

    # file5 = "WN18_1epoch_TransH_pytorch_entity_50dim_batch4800"
    # file6 = "WN18_1epoch_TransH_pytorch_norm_relations_50dim_batch4800"
    # file7 = "WN18_1epoch_TransH_pytorch_hyper_relations_50dim_batch4800"
    # file8 = "WN18\\wordnet-mlj12-test.txt"
    # file9 = "Fb15k_loss_record.txt"
    # transH = TransH(entity_set, relation_set, triple_list, embedding_dim=50, lr=0.005, margin=4.0, norm=1, C=0.25, epsilon=1e-5, valid_triple_list = valid_triple_list)
    # transH.data_initialise()
    # transH.insert_data(file5, file6, file7, file8, file9)
    # # transH.training_run(epochs=500, batch_size=4800, out_file_title="WN18_1epoch_")
    # transH.test_run(filter = False)


    file5 = "FB15k_50epoch_TransH_pytorch_entity_200dim_batch1200"
    file6 = "FB15k_50epoch_TransH_pytorch_norm_relations_200dim_batch1200"
    file7 = "FB15k_50epoch_TransH_pytorch_hyper_relations_200dim_batch1200"
    file8 = "FB15k\\freebase_mtr100_mte100-test.txt"
    file9 = "Fb15k_loss_record.txt"
    transH = TransH(entity_set, relation_set, triple_list, embedding_dim=200, lr=0.001, margin=8.0, norm=1, C=1.0, epsilon=1e-5, valid_triple_list = valid_triple_list)
    transH.data_initialise()
    transH.insert_data(file5, file6, file7, file8, file9)
    # transH.training_run(epochs=50, batch_size=1200, out_file_title="FB15k_50epoch_")
    transH.test_run(filter = False)








