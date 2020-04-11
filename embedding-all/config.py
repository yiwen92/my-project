import os

# GPU 设置
# nvidia-smi
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

pwd_path = os.path.abspath(os.path.dirname(__file__))

MAX_NUM_NEG = 2         # 负采样数目
VOCAB_SIZE = 5000       # 词汇表大小
EMBEDDING_DIM = 3      # 词向量维度
SEMANTIC_DIM = 4       # 语义向量大小
SEQ_LEN = 3             # 序列长度

class Config:
    def __init__(self):
        self.train_samples = os.path.join(pwd_path, "data/train_sample")
        self.vocab = os.path.join(pwd_path, "dict/vocab")
        self.idf = os.path.join(pwd_path, "dict/idf")
        self.stop_words = os.path.join(pwd_path, "dict/stopwords.txt")
        self.place_names = os.path.join(pwd_path, "dict/cities.csv")
        self.func_file = "dict/func.txt"
        self.indus_file = "dict/ind.txt"

conf = Config()