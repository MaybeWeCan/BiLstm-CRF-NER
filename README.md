一、参考

1. 原代码github
https://github.com/wavewangyue/ner

2. logging
git管理文件，如果代码运行结果只是打印在cmd里不保存，将毫无意义，因此需要一个日志框架来保存日志。
https://www.cnblogs.com/pycode/p/logging.html

3. 配置文件类
    添加配置文件类，解耦写死的参数

4. 数据处理，多线程
   （1） python多线程、多进程处理单个（大，超大）文件
        https://blog.csdn.net/weixin_43922901/article/details/109072215
    （2）python 彻底解读多线程与多进程
        https://blog.csdn.net/lzy98/article/details/88819425
    （3）理解Python的协程(Coroutine)
        https://www.jianshu.com/p/84df78d3225a

二、针对代码做的优化

（1）修改数据处理类
    
        a. 原数据处理为一个类在做，对大数据以及硬件内存限制不友好。个人将其解耦成两部分：
	           * prepare.py(多进程处理单文件)
	           * get_batch() (yield机制)

        b. 原代码有意思的地方：padding时的长度按照batch内最长长度处理。
    
（2）embedding

	a.明确指明初始化方式以及初始化公式
```
#sqrt( 6 / embedding_dim )
#refer :https://www.dazhuanlan.com/2019/08/17/5d57727232c29/
r = tf.sqrt(tf.cast(6/embedding_dim,dtype=tf.float32))
embedding_matrix = tf.get_variable("embedding_matrix",shape=[vocab_size_char, embedding_dim],initializer=tf.random_uniform_initializer(minval=-r,maxval=r),dtype=tf.float32)
```

  b.对padding做0 mask( padding对应的embeeding会被训练，变为不为0的东西，这里把他mask掉)

```
#build the raw mask array
raw_mask_array = [[1.]] * padding_id + [[0.]] + [[1.]] * (vocab_size_char - padding_id - 1)

mask_padding_lookup_table = tf.get_variable("mask_padding_lookup_table",initializer=raw_mask_array,dtype=tf.float32,trainable=False)

embeddeding = tf.nn.embedding_lookup(embedding_matrix, self.inputs_seq)

#refer: https://www.dazhuanlan.com/2019/08/17/5d57727232c29/
mask_padding_input = tf.nn.embedding_lookup(mask_padding_lookup_table, self.inputs_seq)

#the mask-padding-zero embedding
embedded = tf.multiply(embeddeding, mask_padding_input) # broadcast
```

(3) tensorboard

  对loss做了tensorboard

(4) debug

    tensorflow自带的debug不支持win系统
    
    pycharm debug代码需要包在if __name__ == __main__里

(5) run_meta_option

  尝试在dev时加入run_meta，以此打印sess每一个节点的占用空间以及时间（会限制一些训练速度，需要时再添加，个人在另一个数据集上去掉了）

（6）predict.py

  加入预测脚本

三、代码

    其他细节不再赘述，代码github仓库：
    
https://github.com/MaybeWeCan/BiLstm-CRF-NER/tree/master
