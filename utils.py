import random
import numpy as np
import os

from Config import Config

def load_vocabulary(path):

    print(" load start")

    w2i = {}
    i2w = {}
    index = 0

    with open(path,"r",encoding="utf-8") as f:
        for line in f.readlines():
            word = line[:-1]
            w2i[word] = index
            i2w[index] = word
            index += 1

    print("vocab from: {}, containing words: {}".format(path, len(i2w)))

    return w2i, i2w


def read_info(info_txt):

    with open(info_txt,"r",encoding="utf-8") as f:
        file_length = f.readline().strip().replace("\n","")

    return file_length


# def get_feedd_dict():


def get_batch(seq_data_path,label_data_path,batch_size,file_length,Configs):


    nums_batchs = int(file_length) // batch_size

    with open(seq_data_path,"r",encoding="utf-8") as sf, \
        open(label_data_path,"r",encoding="utf-8") as lf:


        for batch in range(nums_batchs):

            batch_seq = []
            batch_label = []
            batch_seq_length = []

            for i in range(batch_size):

                seq_line = sf.readline().strip().replace("\n","").split()
                label_line = lf.readline().strip().replace("\n","").split()

                batch_seq.append(seq_line)
                batch_label.append(label_line)
                batch_seq_length.append(len(seq_line))

            # padding
            max_seq_len = max(batch_seq_length)

            for seq in batch_seq:
                seq.extend([Configs.w2i_char["[PAD]"]] * (max_seq_len - len(seq)))

            for seq in batch_label:
                seq.extend([Configs.w2i_bio["O"]] * (max_seq_len - len(seq)))

            yield (np.array(batch_seq, dtype="int32"),
                    np.array(batch_seq_length, dtype="int32"),
                    np.array(batch_label, dtype="int32"))


# 原有类，用来对测试集做处理，用上面函数有点麻烦o.o，一般测试集合都不大。。
class DataProcessor_LSTM(object):
    def __init__(self,
                 input_seq_path,
                 output_seq_path,
                 w2i_char,
                 w2i_bio,
                 shuffling=False):

        inputs_seq = []
        with open(input_seq_path, "r", encoding="utf-8") as f:
            for line in f.read().strip().split("\n"):
                seq = [w2i_char[word] if word in w2i_char else w2i_char["[UNK]"] for word in line.split(" ")]
                inputs_seq.append(seq)

        outputs_seq = []
        with open(output_seq_path, "r", encoding="utf-8") as f:
            for line in f.read().strip().split("\n"):
                seq = [w2i_bio[word] for word in line.split(" ")]
                outputs_seq.append(seq)

        assert len(inputs_seq) == len(outputs_seq)
        assert all(len(input_seq) == len(output_seq) for input_seq, output_seq in zip(inputs_seq, outputs_seq))

        self.w2i_char = w2i_char
        self.w2i_bio = w2i_bio
        self.inputs_seq = inputs_seq
        self.outputs_seq = outputs_seq
        self.ps = list(range(len(inputs_seq)))
        self.shuffling = shuffling
        if shuffling: random.shuffle(self.ps)
        self.pointer = 0
        self.end_flag = False
        print("DataProcessor load data num: " + str(len(inputs_seq)), "shuffling:", shuffling)

    def refresh(self):
        if self.shuffling: random.shuffle(self.ps)
        self.pointer = 0
        self.end_flag = False

    def get_batch(self, batch_size):
        inputs_seq_batch = []
        inputs_seq_len_batch = []
        outputs_seq_batch = []

        while (len(inputs_seq_batch) < batch_size) and (not self.end_flag):
            p = self.ps[self.pointer]
            inputs_seq_batch.append(self.inputs_seq[p].copy())
            inputs_seq_len_batch.append(len(self.inputs_seq[p]))
            outputs_seq_batch.append(self.outputs_seq[p].copy())
            self.pointer += 1
            if self.pointer >= len(self.ps): self.end_flag = True

        # need
        max_seq_len = max(inputs_seq_len_batch)
        for seq in inputs_seq_batch:
            seq.extend([self.w2i_char["[PAD]"]] * (max_seq_len - len(seq)))
        for seq in outputs_seq_batch:
            seq.extend([self.w2i_bio["O"]] * (max_seq_len - len(seq)))

        return (np.array(inputs_seq_batch, dtype="int32"),
                np.array(inputs_seq_len_batch, dtype="int32"),
                np.array(outputs_seq_batch, dtype="int32"))



######################################
####### extract_kvpairs_by_bio #######
######################################

def extract_kvpairs_in_bio(bio_seq, word_seq):
    assert len(bio_seq) == len(word_seq)
    pairs = set()
    pre_bio = "O"
    v = ""
    for i, bio in enumerate(bio_seq):
        if (bio == "O"):
            if v != "": pairs.add((pre_bio[2:], v))
            v = ""
        elif (bio[0] == "B"):
            if v != "": pairs.add((pre_bio[2:], v))
            v = word_seq[i]
        elif (bio[0] == "I"):
            if (pre_bio[0] == "O") or (pre_bio[2:] != bio[2:]):
                if v != "": pairs.add((pre_bio[2:], v))
                v = ""
            else:
                v += word_seq[i]
        pre_bio = bio
    if v != "": pairs.add((pre_bio[2:], v))
    return pairs

def extract_kvpairs_in_bioes(bio_seq, word_seq, attr_seq):
    assert len(bio_seq) == len(word_seq) == len(attr_seq)
    pairs = set()
    v = ""
    for i in range(len(bio_seq)):
        word = word_seq[i]
        bio = bio_seq[i]
        attr = attr_seq[i]
        if bio == "O":
            v = ""
        elif bio == "S":
            v = word
            pairs.add((attr, v))
            v = ""
        elif bio == "B":
            v = word
        elif bio == "I":
            if v != "":
                v += word
        elif bio == "E":
            if v != "":
                v += word
                pairs.add((attr, v))
            v = ""
    return pairs


############################
####### cal_f1_score #######
############################

def cal_f1_score(preds, golds):
    assert len(preds) == len(golds)
    p_sum = 0
    r_sum = 0
    hits = 0
    for pred, gold in zip(preds, golds):
        p_sum += len(pred)
        r_sum += len(gold)
        for label in pred:
            if label in gold:
                hits += 1
    p = hits / p_sum if p_sum > 0 else 0
    r = hits / r_sum if r_sum > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return p, r, f1





            


