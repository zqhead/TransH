'''
这一版使用pytorch的自动微分来对模型进行改造

'''

import torch
import torch.optim as optim
import torch.nn.functional as F

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

            h_ = entity2id[triple[0]]
            t_ = entity2id[triple[1]]
            r_ = relation2id[triple[2]]

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

    return entity_set, relation_set, triple_list

def norm_l1(h, r, t):
    return np.sum(np.fabs(h + r - t))




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
        self.norm_relations = {}
        self.hyper_relations = {}
        self.C = C
        self.epsilon = epsilon

    def data_initialise(self):
        entityVectorList = {}
        relationNormVectorList = {}
        relationHyperVectorList = {}
        device = "cpu"
        for entity in self.entities:
            entity_vector = torch.Tensor(self.dimension).uniform_(-6.0 / np.sqrt(self.dimension), 6.0 / np.sqrt(self.dimension))
            entityVectorList[entity] = entity_vector.requires_grad_(True)

        for relation in self.relations:
            relation_norm_vector = torch.Tensor(self.dimension).uniform_(-6.0 / np.sqrt(self.dimension), 6.0 / np.sqrt(self.dimension))
            relation_hyper_vector = torch.Tensor(self.dimension).uniform_(-6.0 / np.sqrt(self.dimension), 6.0 / np.sqrt(self.dimension))


            relationNormVectorList[relation] = relation_norm_vector.requires_grad_(True)
            relationHyperVectorList[relation] = relation_hyper_vector.requires_grad_(True)

        self.entities = entityVectorList
        self.norm_relations = relationNormVectorList
        self.hyper_relations = relationHyperVectorList


    def training_run(self, epochs=100, nbatches=100):

        batch_size = int(len(self.triples) / nbatches)
        print("batch size: ", batch_size)
        for epoch in range(epochs):
            start = time.time()
            self.loss = 0.0
            # Normalise the embedding of the entities to 1
            # for entity in self.entities:
            #     self.entities[entity] = self.normalization(self.entities[entity])

            for batch in range(nbatches):
                batch_samples = random.sample(self.triples, batch_size)

                Tbatch = []
                for sample in batch_samples:
                    corrupted_sample = copy.deepcopy(sample)
                    pr = np.random.random(1)[0]
                    p = relation_tph[corrupted_sample[2]] / (
                                relation_tph[corrupted_sample[2]] + relation_hpt[corrupted_sample[2]])
                    '''
                    这里关于p的说明 tph 表示每一个头结对应的平均尾节点数 hpt 表示每一个尾节点对应的平均头结点数
                    当tph > hpt 时 更倾向于替换头 反之则跟倾向于替换尾实体

                    举例说明 
                    在一个知识图谱中，一共有10个实体 和n个关系，如果其中一个关系使两个头实体对应五个尾实体，
                    那么这些头实体的平均 tph为2.5，而这些尾实体的平均 hpt只有0.4，
                    则此时我们更倾向于替换头实体，
                    因为替换头实体才会有更高概率获得正假三元组，如果替换头实体，获得正假三元组的概率为 8/9 而替换尾实体获得正假三元组的概率只有 5/9
                    '''
                    if pr < p:
                        # change the head entity
                        corrupted_sample[0] = random.sample(self.entities.keys(), 1)[0]
                        while corrupted_sample[0] == sample[0]:
                            corrupted_sample[0] = random.sample(self.entities.keys(), 1)[0]
                    else:
                        # change the tail entity
                        corrupted_sample[1] = random.sample(self.entities.keys(), 1)[0]
                        while corrupted_sample[1] == sample[1]:
                            corrupted_sample[1] = random.sample(self.entities.keys(), 1)[0]

                    if (sample, corrupted_sample) not in Tbatch:
                        Tbatch.append((sample, corrupted_sample))

                self.update_triple_embedding(Tbatch)
            end = time.time()
            print("epoch: ", epoch, "cost time: %s" % (round((end - start), 3)))
            print("running loss: ", self.loss)

        with codecs.open("entity_" + str(self.dimension) + "dim_batch" + str(batch_size), "w") as f1:

            for e in self.entities:
                f1.write(e + "\t")
                f1.write(str(list(self.entities[e])))
                f1.write("\n")

        with codecs.open("relation_norm_" + str(self.dimension) + "dim_batch" + str(batch_size), "w") as f2:
            for r in self.norm_relations:
                f2.write(r + "\t")
                f2.write(str(list(self.norm_relations[r])))
                f2.write("\n")

        with codecs.open("relation_hyper_" + str(self.dimension) + "dim_batch" + str(batch_size), "w") as f3:
            for r in self.hyper_relations:
                f3.write(r + "\t")
                f3.write(str(list(self.hyper_relations[r])))
                f3.write("\n")


    def normalization(self, vector):
        v = vector / torch.sum(torch.square(vector))
        return v.requires_grad_(True)

    def norm_l2(self, h, r_norm, r_hyper, t):
        return torch.norm(h - r_norm.dot(h)*r_norm + r_hyper -(t - r_norm.dot(t)*r_norm))


    # 知乎上询问过清华的大佬对于软约束项的建议 模长约束对结果收敛有影响，但是正交约束影响很小所以模长约束保留，正交约束可以不加
    def scale_entity(self, vector):
        return torch.relu(torch.sum(vector**2) - 1)

    def orthogonality(self, norm, hyper):
        return np.dot(norm, hyper)**2/np.linalg.norm(hyper)**2 - self.epsilon**2

    def update_triple_embedding(self, Tbatch):

        for correct_sample, corrupted_sample in Tbatch:
            correct_head = self.entities[correct_sample[0]]
            correct_tail  = self.entities[correct_sample[1]]
            relation_norm = self.norm_relations[correct_sample[2]]
            relation_hyper = self.hyper_relations[correct_sample[2]]

            corrupted_head = self.entities[corrupted_sample[0]]
            corrupted_tail = self.entities[corrupted_sample[1]]

            # # calculate the distance of the triples
            # correct_distance = self.norm_l2(correct_head, relation_norm, relation_hyper, correct_tail)
            # corrupted_distance = self.norm_l2(corrupted_head, relation_norm, relation_hyper, corrupted_tail)

            opt1 = optim.SGD([correct_head], lr=0.01)
            opt2 = optim.SGD([correct_tail], lr=0.01)
            opt3 = optim.SGD([relation_norm], lr=0.01)
            opt4 = optim.SGD([relation_hyper], lr=0.01)

            if correct_sample[0] == corrupted_sample[0]:
                opt5 = optim.SGD([corrupted_tail], lr=0.01)
                correct_distance = self.norm_l2(correct_head, relation_norm, relation_hyper, correct_tail)
                corrupted_distance = self.norm_l2(correct_head, relation_norm, relation_hyper, corrupted_tail)
                scale = self.scale_entity(correct_head) + self.scale_entity(correct_tail) + self.scale_entity(corrupted_tail)

            else:
                opt5 = optim.SGD([corrupted_head], lr=0.01)
                correct_distance = self.norm_l2(correct_head, relation_norm, relation_hyper, correct_tail)
                corrupted_distance = self.norm_l2(corrupted_head, relation_norm, relation_hyper, correct_tail)
                scale = self.scale_entity(correct_head) + self.scale_entity(correct_tail) + self.scale_entity(
                    corrupted_head)

            opt1.zero_grad()
            opt2.zero_grad()
            opt3.zero_grad()
            opt4.zero_grad()
            opt5.zero_grad()

            loss = F.relu(self.margin + correct_distance - corrupted_distance) + self.C * scale
            loss.backward()
            self.loss += loss
            opt1.step()
            opt2.step()
            opt3.step()
            opt4.step()
            opt5.step()


            # normalising these new embedding vector, instead of normalising all the embedding together
            self.entities[correct_sample[0]] = correct_head
            self.entities[correct_sample[1]] = correct_tail
            if correct_sample[0] == corrupted_sample[0]:
                # if corrupted_triples replace the tail entity, update the tail entity's embedding
                self.entities[corrupted_sample[1]] = corrupted_tail
            elif correct_sample[1] == corrupted_sample[1]:
                # if corrupted_triples replace the head entity, update the head entity's embedding
                self.entities[corrupted_sample[0]] = corrupted_head
            # the paper mention that the relation's embedding don't need to be normalised
            self.norm_relations[correct_sample[2]] = relation_norm
            self.hyper_relations[correct_sample[2]] = relation_hyper

if __name__ == '__main__':
    file1 = "FB15k\\"
    entity_set, relation_set, triple_list = data_loader(file1)

    transH = TransH(entity_set, relation_set, triple_list, embedding_dim=50, lr=0.01, margin=1.0, norm=1)
    transH.data_initialise()
    transH.training_run()



















