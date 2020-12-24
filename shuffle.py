'''
xxx.py input_file output_file
'''
import os, sys
import random

# 得到line的pos
def get_line_offset(f):
    lines_start_offset=list()
    f.seek(0)
    lines_start_offset.append(f.tell())
    line = f.readline()
    while line:
        # line=line.strip()
        lines_start_offset.append(f.tell())
        line = f.readline()
    return lines_start_offset

def rewrite_file(f_in, f_out, lines_start_offset):
    for i in range(len(lines_start_offset)):
        f_in.seek(lines_start_offset[i], 0)
        line=f_in.readline()
        f_out.write(line)


def shuffle_file(path1,path2,w_path1,w_path2):

    f_train = open(path1, 'r', encoding='utf-8')
    f_label = open(path2, 'r', encoding='utf-8')

    f_train_out = open(w_path1, 'w', encoding='utf-8')
    f_label_out = open(w_path2, 'w', encoding='utf-8')

    train_lines_start_offset = get_line_offset(f_train)
    label_lines_start_offset = get_line_offset(f_label)

    index = list(range(len(train_lines_start_offset)))
    random.shuffle(index)

    train_lines_start_offset = list(map(lambda i: train_lines_start_offset[i], index))
    label_lines_start_offset = list(map(lambda i: label_lines_start_offset[i], index))

    rewrite_file(f_train, f_train_out, train_lines_start_offset)
    rewrite_file(f_label, f_label_out, label_lines_start_offset)

    f_train.close()
    f_label.close()
    f_train_out.close()
    f_label_out.close()


if __name__ == "__main__":

    path1 = "./data_2/train/prepare_seq.txt"
    path2 = "./data_2/train/prepare_label.txt"

    w_path1 = "./data_2/train/shuffle_seq.txt"
    w_path2 = "./data_2/train/shuffle_label.txt"

    f_train = open(path1, 'r', encoding='utf-8')
    f_label = open(path2, 'r', encoding='utf-8')

    f_train_out = open(w_path1, 'w', encoding='utf-8')
    f_label_out = open(w_path2, 'w', encoding='utf-8')

    train_lines_start_offset = get_line_offset(f_train)
    label_lines_start_offset = get_line_offset(f_label)


    index = list(range(len(train_lines_start_offset)))
    random.shuffle(index)

    train_lines_start_offset = list(map(lambda i:train_lines_start_offset[i],index))
    label_lines_start_offset = list(map(lambda i:label_lines_start_offset[i],index))


    rewrite_file(f_train, f_train_out, train_lines_start_offset)
    rewrite_file(f_label, f_label_out, label_lines_start_offset)

    f_train.close()
    f_label.close()
    f_train_out.close()
    f_label_out.close()