
import os
from collections import Counter
import random

from multiprocessing import Pool
from utils import load_vocabulary
from Config import Config

'''
    多线程的尝试使用,此处不考虑内存问题:
        多处理，单一写入
'''

def safe_readline(f):
    pos = f.tell()
    while True:
        try:
            return f.readline()
        except UnicodeDecodeError:
            pos -= 1
            f.seek(pos)

def process_line_to_index(line_list,word_to_index,islabel=True):

    seq = []

    for word in line_list:

        if not islabel:
            index = word_to_index.get(word,word_to_index["UNK"])
        else:
            index = word_to_index.get(word, word_to_index["O"])
        seq.append(index)

    return seq


def process_single_file(path,min_count = 2):

    assert os.path.exists(path)

    results_seq = []
    results_label = []


    label_vocab = set()

    with open(path, 'r', encoding="utf-8") as f:
        line = f.readline()

        seq = []
        label = []
        vocab_counter = Counter()

        while line:

            if(line != "\n"):
                line = line.strip().split("\t")
                seq.append(line[0])
                label.append(line[1])
                label_vocab.add(line[1])

            else:
                tmp = Counter(seq)
                vocab_counter+=tmp

                results_seq.append(seq)
                results_label.append(label)
                seq = []
                label = []

            line = f.readline()

    # 转换词库

    char_vocab = []


    for vocab in vocab_counter.items():

        if vocab[1] < min_count:
            continue
        else:
            char_vocab.append(vocab[0])


    return results_seq,results_label,char_vocab,label_vocab


def save_results_to_file(results, write_path):

    # write into file
    with open(write_path,"w") as wf:

        for single_seq in results:
            # "|".join(str(i) for i in data)
            wrtie_line = " ".join(str(i) for i in single_seq)
            wrtie_line += "\n"
            wf.write(wrtie_line)

def save_data_info(seq_results,info_txt_file):
    seq_length = len(seq_results)

    with open(info_txt_file,"w",encoding="utf-8") as f:
        f.write(str(seq_length))

def save_data_vocab(vocabs,vocab_file):

    with open(vocab_file,"w",encoding="utf-8") as f:

        for vocab in vocabs:
            f.write(vocab + "\n")

if __name__ == '__main__':


    Configs = Config()

    # create data_dir

    for dir in [Configs.vocab_dir,
                Configs.train_dir,
                Configs.dev_dir,
                Configs.prepare_dir]:

        if os.path.exists(dir):
            continue
        else:
            os.mkdir(dir)

    # process data
    results_seq,results_label,char_vocab,label_vocab = process_single_file(Configs.inital_data)

    label_vocab = list(label_vocab)


    # 填下表
    special_char = ["PAD","UNK"]
    char_vocab = special_char + char_vocab

    # 记录词库和索引库
    save_data_vocab(char_vocab, Configs.char_vocab_path)
    save_data_vocab(label_vocab,Configs.label_vocab_path)

    w2i_char, i2w_char = load_vocabulary(Configs.char_vocab_path)
    w2i_bio, i2w_bio = load_vocabulary(Configs.label_vocab_path)

    # 索引化
    index_seq = []
    index_label = []

    for line,label in zip(results_seq,results_label):
        index_seq.append(process_line_to_index(line,w2i_char,islabel=False))
        index_label.append(process_line_to_index(label,w2i_bio))




    # 输出处理好的数据
    save_results_to_file(index_seq,Configs.write_seq_path)
    save_results_to_file(index_label,Configs.write_label_path)


    # 输出训练数据

    index = list(range(len(index_seq)))
    random.shuffle(index)

    index_seq = list(map(lambda i: index_seq[i], index))
    index_label = list(map(lambda i: index_label[i], index))

    # 输出train
    save_results_to_file(index_seq[:-Configs.dev_number],Configs.train_seq_path)
    save_results_to_file(index_label[:-Configs.dev_number],Configs.train_label_path)

    save_data_info(index_seq[:-Configs.dev_number], Configs.info_txt)

    # 输出dev数据
    save_results_to_file(index_seq[-Configs.dev_number:],Configs.test_seq_path)
    save_results_to_file(index_label[-Configs.dev_number:],Configs.test_label_path)

    print("Prepare data Successfully!")


