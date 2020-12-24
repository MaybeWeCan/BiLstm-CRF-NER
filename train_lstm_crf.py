import logging
import tensorflow as tf
import numpy as np
import os
import math

from model_lstm_crf import MyModel
from utils import DataProcessor_LSTM as DataProcessor
from utils import load_vocabulary
from utils import extract_kvpairs_in_bio
from utils import cal_f1_score
from utils import get_batch
from utils import read_info
from Config import Config
from shuffle import get_line_offset
from shuffle import rewrite_file
from shuffle import shuffle_file



# model, sess, total_step, data_processor_valid, batch_iter
def valid(model, sess, train_global_step, dev_batch_iter, i2w_bio, i2w_char):

    preds_kvpair = []
    golds_kvpair = []

    for dev_batche in dev_batch_iter:

        inputs_seq_batch,inputs_seq_len_batch,outputs_seq_batch = dev_batche[0],dev_batche[1],dev_batche[2]

        feed_dict = {
            model.inputs_seq: inputs_seq_batch,
            model.inputs_seq_len: inputs_seq_len_batch,
            model.outputs_seq: outputs_seq_batch
        }

        # 将配置信息和记录运行信息的proto传入运行的过程，从而记录运行时每一个节点的时间、空间开销信息
        preds_seq_batch,test_global_step,test_loss_sum = sess.run([model.outputs,
                                                  model.global_step,
                                                  train_merged_summary_op],
                                                 feed_dict)

        # preds_seq_batch= sess.run([model.outputs],feed_dict)



        # tag如果重复会报错
        # train_writer.add_run_metadata(run_metadata, 'step%03d' % train_global_step)

        for pred_seq, gold_seq, input_seq, l in zip(preds_seq_batch,
                                                    outputs_seq_batch,
                                                    inputs_seq_batch,
                                                    inputs_seq_len_batch):

            pred_seq = [i2w_bio[i] for i in pred_seq[:l]]
            gold_seq = [i2w_bio[i] for i in gold_seq[:l]]
            char_seq = [i2w_char[i] for i in input_seq[:l]]

            pred_kvpair = extract_kvpairs_in_bio(pred_seq, char_seq)
            gold_kvpair = extract_kvpairs_in_bio(gold_seq, char_seq)

            preds_kvpair.append(pred_kvpair)
            golds_kvpair.append(gold_kvpair)

    test_writer.add_summary(test_loss_sum, train_global_step)

    # 计算f1 score
    p, r, f1 = cal_f1_score(preds_kvpair, golds_kvpair)
    logger.info("Valid P/R/F1: {} / {} / {}".format(round(p * 100, 2), round(r * 100, 2), round(f1 * 100, 2)))

    return p, r, f1


if __name__ == "__main__":


    Configs = Config()

    if os.path.exists(Configs.log_file_path):
        os.remove(Configs.log_file_path)

    # 清理原有的


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
    fhlr = logging.FileHandler(Configs.log_file_path)
    fhlr.setFormatter(formatter)

    logger.addHandler(chlr)
    logger.addHandler(fhlr)


    logger.info("loading vocab...")
    w2i_char, i2w_char = load_vocabulary(Configs.char_vocab_path)
    w2i_bio, i2w_bio = load_vocabulary(Configs.label_vocab_path)


    pad_index = w2i_char.get("PAD",-1)
    Configs.padding_id = pad_index

    if pad_index == -1:
        logger.info("PAD is not in the vocab")
        exit(1)

    Configs.w2i_char = w2i_char
    Configs.w2i_bio = w2i_bio


    logger.info("read info.txt")
    file_length = read_info(Configs.info_txt)



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

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=Configs.max_to_keep)

        train_writer = tf.summary.FileWriter(Configs.train_tenboard_dir, sess.graph)
        test_writer = tf.summary.FileWriter(Configs.val_tenboard_dir)

        # tensorboard
        tf.summary.scalar("loss", model.loss)
        train_merged_summary_op = tf.summary.merge_all()

        losses = []
        best_f1 = 0
        total_step = 0

        for epoche in range(Configs.epoches):

            logging.info(" shuffle file!!!")
            # 随机打乱数据
            shuffle_file(Configs.train_seq_path,
                         Configs.train_label_path,
                         Configs.train_shuffle_seq_path,
                         Configs.train_shuffle_label_path)

            # 文件名改变
            os.remove(Configs.train_seq_path)
            os.remove(Configs.train_label_path)

            os.rename(Configs.train_shuffle_seq_path,
                      Configs.train_seq_path)

            os.rename(Configs.train_shuffle_label_path,
                      Configs.train_label_path)

            logging.info(" shuffle end!!!")

            train_batch_iter = get_batch(Configs.train_seq_path,
                                   Configs.train_label_path,
                                   Configs.batch_size,
                                   file_length,
                                   Configs)

            print_data_step = 0
            batch_step = 0

            for (inputs_seq_batch, inputs_seq_len_batch, outputs_seq_batch) in train_batch_iter:

                if batch_step == print_data_step:
                    logger.info("###### shape of a batch #######")
                    logger.info("input_seq: " + str(inputs_seq_batch.shape))
                    logger.info("input_seq_len: " + str(inputs_seq_len_batch.shape))
                    logger.info("output_seq: " + str(outputs_seq_batch.shape))
                    logger.info("###### preview a sample #######")
                    logger.info("input_seq:" + " ".join([i2w_char[i] for i in inputs_seq_batch[0]]))
                    logger.info("input_seq_len :" + str(inputs_seq_len_batch[0]))
                    logger.info("output_seq: " + " ".join([i2w_bio[i] for i in outputs_seq_batch[0]]))
                    logger.info("###############################")

                logging.info("loss compute start !!!")

                feed_dict = {
                    model.inputs_seq: inputs_seq_batch,
                    model.inputs_seq_len: inputs_seq_len_batch,
                    model.outputs_seq: outputs_seq_batch
                }

                # 这次run要占用很大内存,能从4个飙升到11G多，然后再回来
                # 4.8 G 飙升到 15.8G
                loss, _, global_steps= sess.run([model.loss,
                                                 model.train_op,
                                                 model.global_step],
                                                feed_dict)

                losses.append(loss)
                batch_step += 1
                total_step+= 1

                # save model
                if batch_step % Configs.save_batch == 0:

                    logger.info("train loss tensorboard visual")

                    _,train_loss_sum = sess.run([model.loss,
                                               train_merged_summary_op],
                                              feed_dict)

                    train_writer.add_summary(train_loss_sum, global_steps)


                    dev_batch_iter = get_batch(Configs.test_seq_path,
                                               Configs.test_label_path,
                                               Configs.batch_size,
                                               Configs.dev_number,
                                               Configs)

                    logger.info("")
                    logger.info("Epoches: {}".format(epoche))
                    logger.info("Batches: {}".format(batch_step))
                    logger.info("Loss: {}".format(sum(losses) / len(losses)))
                    losses = []

                    ckpt_save_path = Configs.ckpt_save_dir+"model.ckpt.batch{}".format(batch_step)

                    logger.info("Path of ckpt: {}".format(ckpt_save_path))


                    p, r, f1 = valid(model,sess,global_steps,dev_batch_iter,i2w_bio,i2w_char)

                    if (f1 - best_f1) > Configs.threshold_save:
                        logger.info("Save model,f1 is {},best f1 is{},globa_step is {}".format(f1,best_f1,global_steps))
                        saver.save(sess, ckpt_save_path, global_step=global_steps)
                        best_f1 = f1

    logger.info("final best f1 is {}".format(best_f1))

    train_writer.close()
    test_writer.close()
