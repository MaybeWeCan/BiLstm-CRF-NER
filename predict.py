import tensorflow as tf
import numpy as np

from Config import Config
from utils import load_vocabulary

def clean_data(data):

    # clean and repalce the data

    return data

def prepare_data(datasets,w2i_char):
    '''
        The form of datasets is [,,]
    '''

    output_datasets = []
    output_length = []

    for data in datasets:

        # It is str
        data = clean_data(data)

        seq_line = []

        for char in data:

            try:
                index = w2i_char[char]
            except:
                index = w2i_char["UNK"]

            seq_line.append(index)


        output_datasets.append(seq_line)
        output_length.append(len(seq_line))

        max_seq_len = max(output_length)

        # Padding,利用了python的那个特性
        for seq in output_datasets:
            seq.extend([w2i_char["PAD"]] * (max_seq_len - len(seq)))


    return (np.array(output_datasets, dtype="int32"),
                    np.array(output_length, dtype="int32"))



def index_to_label(output_indexs,i2w_bio):

    result = []

    # 其输出不可能需要改
    output_indexs = output_indexs[0].tolist()

    for index_nary in output_indexs:

        # print(index_nary)

        label_list = []
        for index in index_nary:
            label_list.append(i2w_bio[index])

        result.append(label_list)

    return result




if __name__ == "__main__":

    Configs = Config()

    # read label vocab
    w2i_char, i2w_char = load_vocabulary(Configs.char_vocab_path)
    w2i_bio, i2w_bio = load_vocabulary(Configs.label_vocab_path)

    # 输入一堆句子
    data = ["李华爱北京天安门"]

    # 句子转换格式
    predict_datasets, predict_length= prepare_data(data,w2i_char)



    # 加载模型

    with tf.Session() as sess:

        saver = tf.train.import_meta_graph(Configs.ckpt_save_dir+'model.ckpt.batch750-10176.meta')
        saver.restore(sess, tf.train.latest_checkpoint(Configs.ckpt_save_dir))

        g = tf.get_default_graph()

        # tensor_name_list = [tensor.name for tensor in g.as_graph_def().node]
        #
        # for tensor_name in tensor_name_list:
        #     print(tensor_name, '\n')

        prep_ouput = g.get_tensor_by_name(name='projection/cond_2/Merge:0')
        input_data = g.get_tensor_by_name(name="inputs_seq:0")
        input_length = g.get_tensor_by_name(name="inputs_seq_len:0")

        feed_dict = {
            input_data:predict_datasets,
            input_length:predict_length
        }

        output = sess.run([prep_ouput],feed_dict=feed_dict)


        label_output = index_to_label(output, i2w_bio)

        print(data)
        print(label_output)

    # 结果输出




