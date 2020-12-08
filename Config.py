class Config:
    def __init__(self):


        self.w2i_char = {}
        self.w2i_bio = {}

        ''' Prepaer train data '''
        self.char_vocab_path = "./data/vocab_char.txt"
        self.label_vocab_path = "./data/vocab_bioattr.txt"

        self.seq_path = "./data/train/input.seq.char"
        self.label_path = "./data/train/output.seq.bioattr"

        self.write_seq_path = "./data/train/output_seq.txt"
        self.write_label_path = "./data/train/output_label.txt"

        self.info_txt = "info.txt"
        self.workers = 4

        ''' Prepaer test data '''

        self.test_seq_path = "./data/test/input.seq.char"
        self.test_label_path = "./data/test/output.seq.bioattr"


        ''' Train model'''
        self.log_file_path = "./ckpt/train_lstm_crf.log"
        self.train_tenboard_dir = './tensorboard/logs/train'
        self.val_tenboard_dir = './tensorboard/logs/val'
        self.max_to_keep = 5
        self.batch_size = 32
        self.epoches = 20

        self.save_batch = 2


        ''' Model '''
        self.embedding_dim = 300
        self.hidden_dim = 300
        # self.vocab_size_char = -1
        # self.vocab_size_bio = -1
        self.use_crf = True

        self.ckpt_save_dir = "./ckpt/"

