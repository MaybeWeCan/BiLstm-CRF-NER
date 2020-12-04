import logging
import tensorflow as tf
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



Configs = Config()

if os.path.exists(Configs.log_file_path):
    os.remove(Configs.log_file_path)

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

model = MyModel(embedding_dim=Configs.embedding_dim,
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

tf_config = tf.ConfigProto(allow_soft_placement=True)
tf_config.gpu_options.allow_growth = True



with tf.Session(config=tf_config) as sess:

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=Configs.max_to_keep)
    

    losses = []
    best_f1 = 0

    for epoche in range(Configs.epoches):

        batch_iter = get_batch(Configs.write_seq_path,
                               Configs.write_label_path,
                               Configs.batch_size,
                               file_length,
                               Configs)

        batch_step = 0

        for (inputs_seq_batch, inputs_seq_len_batch, outputs_seq_batch) in batch_iter:


        
            feed_dict = {
                model.inputs_seq: inputs_seq_batch,
                model.inputs_seq_len: inputs_seq_len_batch,
                model.outputs_seq: outputs_seq_batch
            }

            if batch_step == 0:
                logger.info("###### shape of a batch #######")
                logger.info("input_seq: " + str(inputs_seq_batch.shape))
                logger.info("input_seq_len: " + str(inputs_seq_len_batch.shape))
                logger.info("output_seq: " + str(outputs_seq_batch.shape))
                logger.info("###### preview a sample #######")
                logger.info("input_seq:" + " ".join([i2w_char[i] for i in inputs_seq_batch[0]]))
                logger.info("input_seq_len :" + str(inputs_seq_len_batch[0]))
                logger.info("output_seq: " + " ".join([i2w_bio[i] for i in outputs_seq_batch[0]]))
                logger.info("###############################")


            loss, _ = sess.run([model.loss, model.train_op], feed_dict)

            # print("loss start !!!")
            # loss = tf.Print(model.loss, [loss], message="loss")


            # losses.append(loss)
            batch_step += 1


            def valid(data_processor, max_batches=None, batch_size=1024):
                preds_kvpair = []
                golds_kvpair = []
                batches_sample = 0

                while True:
                    (inputs_seq_batch,
                     inputs_seq_len_batch,
                     outputs_seq_batch) = data_processor.get_batch(batch_size)

                    feed_dict = {
                        model.inputs_seq: inputs_seq_batch,
                        model.inputs_seq_len: inputs_seq_len_batch,
                        model.outputs_seq: outputs_seq_batch
                    }

                    preds_seq_batch = sess.run(model.outputs, feed_dict)

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

                    if data_processor.end_flag:
                        data_processor.refresh()
                        break

                    batches_sample += 1
                    if (max_batches is not None) and (batches_sample >= max_batches):
                        break

                p, r, f1 = cal_f1_score(preds_kvpair, golds_kvpair)

                logger.info("Valid Samples: {}".format(len(preds_kvpair)))
                logger.info("Valid P/R/F1: {} / {} / {}".format(round(p*100, 2), round(r*100, 2), round(f1*100, 2)))

                return (p, r, f1)


            # save model
            if batch_step % Configs.save_batch == 0:
                logger.info("")
                logger.info("Epoches: {}".format(epoche))
                logger.info("Batches: {}".format(batch_step))
                logger.info("Loss: {}".format(sum(losses) / len(losses)))
                losses = []

                ckpt_save_path = Configs.ckpt_save_dir+"model.ckpt.batch{}".format(batch_step)

                logger.info("Path of ckpt: {}".format(ckpt_save_path))

                # 在这一步骤一直超过内存
                saver.save(sess, ckpt_save_path)

                p, r, f1 = valid(data_processor_valid, max_batches=10)

                if f1 > best_f1:
                    best_f1 = f1
                    logger.info("############# best performance now here ###############")


