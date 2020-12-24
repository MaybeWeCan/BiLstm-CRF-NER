class Config:
    def __init__(self):


        self.w2i_char = {}
        self.w2i_bio = {}

        ''' Prepaer train data '''

        self.dev_number = 5000

        self.data_dir = "./data_2/"
        self.vocab_dir = self.data_dir + "vocab/"
        self.train_dir = self.data_dir + "train/"
        self.dev_dir = self.data_dir + "dev/"
        self.prepare_dir = self.data_dir + "prepare_data/"

        self.inital_data = self.data_dir+"dh_msra.txt"

        self.char_vocab_path = self.vocab_dir + "char_vocab.txt"
        self.label_vocab_path = self.vocab_dir + "label_vocab.txt"

        self.write_seq_path = self.prepare_dir + "prepare_seq.txt"
        self.write_label_path = self.prepare_dir + "prepare_label.txt"


        self.train_seq_path = self.train_dir + "train_seq.txt"
        self.train_label_path = self.train_dir + "train_label.txt"

        self.train_shuffle_seq_path = self.train_dir + "shuffle_train_seq.txt"
        self.train_shuffle_label_path = self.train_dir + "shuffle_train_label.txt"

        self.test_seq_path = self.dev_dir + "dev_seq.txt"
        self.test_label_path = self.dev_dir + "dev_label.txt"

        self.info_txt = "info.txt"

        self.workers = 4


        ''' Train model'''
        self.log_file_path = "./log/train_lstm_crf.log"
        self.debug_log_file_path = "./log/debug_lstm_crf.log"
        self.train_tenboard_dir = './tensorboard/logs/train'
        self.val_tenboard_dir = './tensorboard/logs/val'
        self.max_to_keep = 5
        self.batch_size = 32
        self.epoches = 20
        # self.test_batch_size = 64

        self.save_batch = 50
        self.threshold_save = 0.01


        ''' Model '''
        self.embedding_dim = 300
        self.hidden_dim = 300
        # self.vocab_size_char = -1
        # self.vocab_size_bio = -1
        self.use_crf = True
        self.padding_id=-1

        self.ckpt_save_dir = "./ckpt/"

