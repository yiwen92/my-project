import os
from absl import flags
import tensorflow as tf

# GPU 设置
# nvidia-smi
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

pwd_path = os.path.abspath(os.path.dirname(__file__))

MAX_NUM_NEG = 8         # 负采样数目
VOCAB_SIZE = 10000       # 词汇表大小
EMBEDDING_DIM = 128      # 词向量维度
SEMANTIC_DIM = 32       # 语义向量大小
SEQ_LEN = 10             # 序列长度

class Config:
    def __init__(self):
        self.learning_rate = 0.01   # 学习率
        self.num_steps = 1000 # estimator 迭代次数
        self.num_epochs = 5  # 总迭代轮次
        self.batch_size = 12  # 每批训练大小
        self.train_valid_ratio = 0.9    # 训练集和测试集的比例
        self.over_write_vocab = False   # 是否重写vocab文件
        self.train_samples = os.path.join(pwd_path, "data/train_sample")
        self.valid_samples = os.path.join(pwd_path, "data/valid_sample")
        self.vocab = os.path.join(pwd_path, "dict/vocab")
        self.idf = os.path.join(pwd_path, "dict/idf")
        self.stop_words = os.path.join(pwd_path, "dict/stopwords.txt")
        self.place_names = os.path.join(pwd_path, "dict/cities.csv")
        self.func_file = "dict/func.txt"
        self.indus_file = "dict/ind.txt"
        self.models_path = "models/"

conf = Config()

flags.DEFINE_bool("use_tpu", False, help="whether to use TPUs")
flags.DEFINE_string("master", default=None, help="master")
flags.DEFINE_integer("num_core_per_host", default=1, help="8 for TPU v2 and v3-8, 16 for larger TPU v3 pod. In the context of GPU training, it refers to the number of GPUs used.")
flags.DEFINE_string("model_dir", default="estimator_model", help="Estimator model_dir.")
flags.DEFINE_integer("iterations", default=1000, help="Number of iterations per repeat loop.")
flags.DEFINE_integer("num_hosts", default=1, help="number of TPU hosts")
flags.DEFINE_integer("max_save", default=0, help="Max number of checkpoints to save. Use 0 to save all.")
flags.DEFINE_integer("save_steps", default=100, help="Save the model for every save_steps. If None, not to save any model.")

FLAGS = tf.app.flags.FLAGS