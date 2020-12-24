一、参考

1. 原代码github
https://github.com/wavewangyue/ner

个人针对代码进行了调整，此处为第二个版本的Base_line

二、运行

1. 修改Config.py

    该文件包含几乎所有参数的设置，根据自己情况调整。

2. 运行prepare_data.py

   主要用于数据文件的转换。
   
3. 运行train_lstm_crf.py
  
   训练模型，保存结果（个人训练的一版——baseline也上传上来了）
   
4. predcit.py
  
   调用模型，预测
   
三、Baseline训练结果

(1) 数据源

  预处理后的数据：
    
    55289 行， 8.74M

  词库：
    
    4346行

  标签：
    
    'I-LOC', 'B-PER', 'O', 'B-LOC', 'I-ORG', 'B-ORG', 'I-PER'


（2）超参数

  dev_number = 5000
  
  batch_size = 32
  
  epoches = 20
  
  embedding_dim = 300
  
  hidden_dim = 300

  模型结构：1层Bilstm + CRF
  
  优化方法： Adma


(3)模型训练loss结果：







(4)最优的F1值：

 best f1 is 0.8893250792692134,globa_step is 10176



(5) 预测结果：





