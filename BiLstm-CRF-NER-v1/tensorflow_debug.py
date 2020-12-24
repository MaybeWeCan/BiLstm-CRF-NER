import logging
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np
import os

from model_lstm_crf import MyModel
from utils import DataProcessor_LSTM as DataProcessor
from utils import load_vocabulary
from utils import extract_kvpairs_in_bio
from utils import cal_f1_score
from utils import get_batch
from utils import read_info
from Config import Config


if __name__ == "__main__":


    Configs = Config()

    if os.path.exists(Configs.debug_log_file_path):
        os.remove(Configs.debug_log_file_path)

    #  创建一个logger,默认为root logger
    logger = logging.getLogger()
    # 调整终端输出级别,设置全局log级别为INFO。注意全局的优先级最高
    logger.setLevel(logging.INFO)

    # 日志格式： 日志时间，日志信息，设置时间输出格式
    formatter = logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    #  创建一个终端输出的handler
    chlr = logging.StreamHandler()
    chlr.setFormatter(formatter)

    #  创建一个文件记录日志的handler
    fhlr = logging.FileHandler(Configs.debug_log_file_path)
    fhlr.setFormatter(formatter)

    logger.addHandler(chlr)
    logger.addHandler(fhlr)


    logger.info("loading vocab...")

    w2i_char, i2w_char = load_vocabulary(Configs.char_vocab_path)

    try:
        pad_index = w2i_char["[PAD]"]
        Configs.padding_id = pad_index
    except:
        pad_index = -1

    if pad_index == -1:
        logger.info("PAD is not in the vocab")
        exit(1)

    # add padding

    w2i_bio, i2w_bio = load_vocabulary(Configs.label_vocab_path)

    Configs.w2i_char = w2i_char
    Configs.w2i_bio = w2i_bio

    logger.info("read info.txt")

    file_length = read_info(Configs.info_txt)

    logging.info("Build test data Processor")

    data_processor_valid = DataProcessor(
        Configs.test_seq_path,
        Configs.test_label_path,
        w2i_char,
        w2i_bio,
        shuffling=True
    )


    logger.info("building model...")

    model = MyModel(Configs.padding_id,
                    embedding_dim=Configs.embedding_dim,
                    hidden_dim=Configs.hidden_dim,
                    vocab_size_char=len(w2i_char),
                    vocab_size_bio=len(w2i_bio),
                    use_crf=Configs.use_crf)


    logger.info("model params:")

    # calculate params
    params_num_all = 0
    for variable in tf.trainable_variables():
        params_num = 1

        # 维度乘积
        for dim in variable.shape:
            params_num *= dim

        params_num_all += params_num
        logger.info("\t {} {} {}".format(variable.name, variable.shape, params_num))

    logger.info("all params num: " + str(params_num_all))

    logger.info("start training...")

    # tf_config = tf.ConfigProto(allow_soft_placement=True)
    # tf_config.gpu_options.allow_growth = True
    # with tf.Session(config=tf_config) as sess:

    logger.info("train_writer create")



    with tf.Session() as sess:

        sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        sess.run(tf.global_variables_initializer())

        losses = []
        best_f1 = 0

        total_step = 0

        for epoche in range(Configs.epoches):

            batch_iter = get_batch(Configs.write_seq_path,
                                   Configs.write_label_path,
                                   16,
                                   file_length,
                                   Configs)

            batch_step = 1

            for (inputs_seq_batch, inputs_seq_len_batch, outputs_seq_batch) in batch_iter:

                if batch_step == 1:
                    logger.info("###### shape of a batch #######")
                    logger.info("input_seq: " + str(inputs_seq_batch.shape))
                    logger.info("input_seq_len: " + str(inputs_seq_len_batch.shape))
                    logger.info("output_seq: " + str(outputs_seq_batch.shape))
                    logger.info("###### preview a sample #######")
                    logger.info("input_seq:" + " ".join([i2w_char[i] for i in inputs_seq_batch[0]]))
                    logger.info("input_seq_len :" + str(inputs_seq_len_batch[0]))
                    logger.info("output_seq: " + " ".join([i2w_bio[i] for i in outputs_seq_batch[0]]))
                    logger.info("###############################")

                print("loss compute start !!!")


                feed_dict = {
                    model.inputs_seq: inputs_seq_batch,
                    model.inputs_seq_len: inputs_seq_len_batch,
                    model.outputs_seq: outputs_seq_batch
                }

                # 这次run要占用很大内存,能从4个飙升到11G多，然后再回来
                # 4.8 G 飙升到 15.8G
                loss, _, global_steps = sess.run([model.loss,
                                                 model.train_op,
                                                 model.global_step],
                                                feed_dict)

                print("loss start !!!")


                losses.append(loss)

                batch_step += 1
                total_step+= 1

