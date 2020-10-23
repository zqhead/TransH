import json
import operator # operator模块输出一系列对应Python内部操作符的函数
import numpy as np
import codecs
import threading
import torch
import time


from TransH_torch import dataloader,entities2id,relations2id


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

    print("Complete load. entity : %d , relation : %d , triple : %d" % (
        len(entity_dic), len(norm_relation), len(triple_list)))

    return entity_dic, norm_relation, hyper_relation, triple_list



class testTransH:
    def __init__(self, entities_dict, norm_relation, hyper_relation, test_triple_list, train_triple_list, valid_triple, filter_triple=False ,norm=1):
        self.entities = entities_dict
        self.norm_relation = norm_relation
        self.hyper_relation = hyper_relation
        self.test_triples = test_triple_list
        self.valid_triples = valid_triple
        self.train_triples = train_triple_list
        self.filter = filter_triple
        self.norm = norm
        self.mean_rank = 0
        self.hit_10 = 0


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
            head_embedding = []
            tail_embedding = []
            norm_relation = []
            hyper_relation = []
            tamp = []

            head_filter = []
            tail_filter = []
            if self.filter:

                for tr in self.train_triples:
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

            for i, entity in enumerate(self.entities.keys()):

                head_triple = [entity, triple[1], triple[2]]
                if self.filter:
                    if head_triple in head_filter:
                        continue
                head_embedding.append(self.entities[head_triple[0]])
                tail_embedding.append(self.entities[head_triple[2]])
                norm_relation.append(self.norm_relation[head_triple[1]])
                hyper_relation.append(self.hyper_relation[head_triple[1]])
                tamp.append(tuple(head_triple))

            distance = self.distance(head_embedding, norm_relation, hyper_relation, tail_embedding)

            for i in range(len(tamp)):
                rank_head_dict[tamp[i]] = distance[i]

            head_embedding = []
            tail_embedding = []
            norm_relation = []
            hyper_relation = []
            tamp = []

            for i, tail in enumerate(self.entities.keys()):

                tail_triple = [triple[0], triple[1], tail]
                if self.filter:
                    if tail_triple in tail_filter:
                        continue
                head_embedding.append(self.entities[tail_triple[0]])
                norm_relation.append(self.norm_relation[tail_triple[1]])
                hyper_relation.append(self.hyper_relation[tail_triple[1]])
                tail_embedding.append(self.entities[tail_triple[2]])
                tamp.append(tuple(tail_triple))

            distance = self.distance(head_embedding, norm_relation, hyper_relation, tail_embedding)
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


    def distance(self, h, r_norm, r_hyper, t):
        head = torch.from_numpy(np.array(h))
        rel_norm = torch.from_numpy(np.array(r_norm))
        rel_hyper = torch.from_numpy(np.array(r_hyper))
        tail = torch.from_numpy(np.array(t))

        h_hyper = head - torch.sum(head * rel_norm, dim = 1, keepdim=True) * rel_norm
        t_hyper = tail - torch.sum(tail * rel_norm, dim = 1, keepdim=True) * rel_norm

        distance = h_hyper + rel_hyper - t_hyper
        score = torch.norm(distance, p=self.norm, dim=1)
        return score.numpy()



if __name__ == "__main__":

    file1 = "WN18\\wordnet-mlj12-test.txt"
    file2 = "WN18\\entity2id.txt"
    file3 = "WN18\\relation2id.txt"
    file4 = "WN18\\wordnet-mlj12-valid.txt"
    entity_set, relation_set, train_triple, valid_triple = dataloader(file1, file2, file3, file4)

    file5 = "WN18_500epoch_TransH_pytorch_entity_50dim_batch1200"
    file6 = "WN18_500epoch_TransH_pytorch_norm_relations_50dim_batch1200"
    file7 = "WN18_500epoch_TransH_pytorch_hyper_relations_50dim_batch1200"
    file8 = "WN18\\wordnet-mlj12-train.txt"

    entity, norm_relation, hyper_relation, test_triple = test_data_loader(file5, file6, file7, file8)

    test = testTransH(entity, norm_relation, hyper_relation, test_triple, train_triple, valid_triple, filter_triple=False, norm=2)
    hit10, mean_rank = test.test_run()
    print("raw entity hits@10: ", hit10)
    print("raw entity meanrank: ",mean_rank)

    # test2 = testTransH(entity, norm_relation, hyper_relation, test_triple, train_triple, valid_triple,
    #                   filter_triple=True, norm=2)
    # filter_hit10, filter_mean_rank = test2.test_run()
    # print("filter entity hits@10: ", filter_hit10)
    # print("filter entity meanrank: ", filter_mean_rank)
