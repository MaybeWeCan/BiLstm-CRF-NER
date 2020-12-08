import tensorflow as tf
import random
import numpy as np

class MyModel(object):
    
    def __init__(self,
                 padding_id,
                 embedding_dim, 
                 hidden_dim, 
                 vocab_size_char, 
                 vocab_size_bio, 
                 use_crf):
        
        self.inputs_seq = tf.placeholder(tf.int32, [None, None], name="inputs_seq")
        self.inputs_seq_len = tf.placeholder(tf.int32, [None], name="inputs_seq_len")
        self.outputs_seq = tf.placeholder(tf.int32, [None, None], name='outputs_seq')

        self.global_step = tf.Variable(0, trainable=False)
        
        with tf.variable_scope('embedding_layer'):

            # build the raw mask array
            raw_mask_array = [[1.]] * padding_id + [[0.]] + [[1.]] * (vocab_size_char - padding_id - 1)

            # sqrt( 6 / embedding_dim )
            # refer :https://www.dazhuanlan.com/2019/08/17/5d57727232c29/
            r = tf.sqrt(tf.cast(6/embedding_dim,dtype=tf.float32))
            embedding_matrix = tf.get_variable("embedding_matrix",
                                               shape=[vocab_size_char, embedding_dim],
                                               initializer=tf.random_uniform_initializer(minval=-r,
                                                                                         maxval=r),
                                               dtype=tf.float32)

            mask_padding_lookup_table = tf.get_variable("mask_padding_lookup_table",
                                                        initializer=raw_mask_array,
                                                        dtype=tf.float32,
                                                        trainable=False)

            embeddeding = tf.nn.embedding_lookup(embedding_matrix, self.inputs_seq)

            # refer: https://www.dazhuanlan.com/2019/08/17/5d57727232c29/
            mask_padding_input = tf.nn.embedding_lookup(mask_padding_lookup_table, self.inputs_seq)

            # the mask-padding-zero embedding
            embedded = tf.multiply(embeddeding, mask_padding_input)  # broadcast

        # 编码
        with tf.variable_scope('encoder'):

            cell_fw = tf.nn.rnn_cell.LSTMCell(hidden_dim)
            cell_bw = tf.nn.rnn_cell.LSTMCell(hidden_dim)

            # 处理中每个序列的实际长度
            ((rnn_fw_outputs, rnn_bw_outputs), (rnn_fw_final_state, rnn_bw_final_state)) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw, 
                cell_bw=cell_bw, 
                inputs=embedded, 
                sequence_length=self.inputs_seq_len,
                dtype=tf.float32
            )

            # 输出相加
            rnn_outputs = tf.add(rnn_fw_outputs, rnn_bw_outputs) # B * S1 * D

        #
        with tf.variable_scope('projection'):

            # 输出概率
            logits_seq = tf.layers.dense(rnn_outputs, vocab_size_bio) # B * S * V
            probs_seq = tf.nn.softmax(logits_seq)
            
            if not use_crf:
                preds_seq = tf.argmax(probs_seq, axis=-1, name="preds_seq") # B * S
            else:

                # 建层
                log_likelihood, self.transition_matrix = tf.contrib.crf.crf_log_likelihood(logits_seq, self.outputs_seq, self.inputs_seq_len)

                # 根据矩阵预测最佳结果
                preds_seq, crf_scores = tf.contrib.crf.crf_decode(logits_seq, self.transition_matrix, self.inputs_seq_len)

        # 最佳结果的预测
        self.outputs = preds_seq
        
        with tf.variable_scope('loss'):

            if not use_crf:

                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_seq, labels=self.outputs_seq) # B * S

                # 屏蔽掉对应无用的填充
                masks = tf.sequence_mask(self.inputs_seq_len, dtype=tf.float32) # B * S

                # 数据转换
                loss = tf.reduce_sum(loss * masks, axis=-1) / tf.cast(self.inputs_seq_len, tf.float32) # B

            else:
                loss = -log_likelihood / tf.cast(self.inputs_seq_len, tf.float32) # B
            
        self.loss = tf.reduce_mean(loss)
        
        with tf.variable_scope('opt'):
            self.train_op = tf.train.AdamOptimizer().minimize(loss,global_step=self.global_step)


    
