import os
from absl import flags
import tensorflow as tf

# GPU 设置
# nvidia-smi
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

pwd_path = os.path.abspath(os.path.dirname(__file__))

MAX_NUM_NEG = 15         # 负采样数目
VOCAB_SIZE = 10000       # 词汇表大小
EMBEDDING_DIM = 128      # 词向量维度
SEMANTIC_DIM = 64       # 语义向量大小
SEQ_LEN = 10             # 序列长度

class Config:
    def __init__(self):
        self.model_type = 'atten'      # 语义编码网名类型: ['nn', 'cnn', 'rnn', 'atten']
        self.num_steps = 10 # estimator 迭代次数
        self.num_epochs = 1  # 总迭代轮次
        self.train_valid_ratio = 0.9    # 训练集和测试集的比例
        self.over_write_vocab = True   # 是否重写vocab文件
        self.vocab = os.path.join(pwd_path, "dict/vocab")
        self.idf = os.path.join(pwd_path, "dict/idf")
        self.stop_words = os.path.join(pwd_path, "dict/stopwords.txt")
        self.place_names = os.path.join(pwd_path, "dict/cities.csv")
        self.func_file = "dict/func.txt"
        self.indus_file = "dict/ind.txt"
        self.entity_file = "dict/entitys.txt"
        self.new_entity_file = "dict/newentity"#"dict/new_entity"
        self.similarity_type = 'cosine'
        self.mu_pos = 1.0
        self.mu_neg = 1.0
        self.C_emb = 1.2
        self.use_max_sim_neg = False
        self.jdtitledesc_file = "data/part-00000"

conf = Config()

flags.DEFINE_integer("batch_size", default=512, help="batch size of every train step")
flags.DEFINE_integer("train_batch_size", default=512, help="batch size of every train step")
flags.DEFINE_float("learning_rate", default=1e-4, help="Maximum learning rate.")
flags.DEFINE_integer("train_steps", default=100, help="Total number of training steps.")
flags.DEFINE_integer("save_steps", default=100, help="Save the model for every save_steps. If None, not to save any model.")
flags.DEFINE_integer("warmup_steps", default=1000, help="Number of steps for linear lr warmup.")
flags.DEFINE_float("clip", default=1.0, help="Gradient clipping value.")
flags.DEFINE_string("original_file", "data/jddata_1000", help="original file ex: jd or resume file")
flags.DEFINE_string("corpus_file", "data/corpus", help="jd or resume file get train text file")
flags.DEFINE_string("train_samples", "data/train_sample", help="train samples json file")
flags.DEFINE_string("valid_samples", "data/valid_sample", help="valid samples json file")
flags.DEFINE_bool("use_tpu", False, help="whether to use TPUs")
flags.DEFINE_string("master", default=None, help="master")
flags.DEFINE_integer("num_core_per_host", default=1, help="8 for TPU v2 and v3-8, 16 for larger TPU v3 pod. In the context of GPU training, it refers to the number of GPUs used.")
flags.DEFINE_string("model_dir", default="estimator_model", help="Estimator model_dir.")
flags.DEFINE_integer("iterations", default=1000, help="Number of iterations per repeat loop.")
flags.DEFINE_integer("num_hosts", default=1, help="number of TPU hosts")
flags.DEFINE_integer("max_save", default=0, help="Max number of checkpoints to save. Use 0 to save all.")
flags.DEFINE_bool("overwrite_data", default=True, help="If False, will use cached data if available.")
flags.DEFINE_string("output_dir", default="proc_data/", help="Output dir for TF records.")
flags.DEFINE_integer("max_seq_length", default=SEQ_LEN, help="Max sequence length : (pretrain flags.seq_len)")
flags.DEFINE_integer("shuffle_buffer", default=2048, help="Buffer size used for shuffle.")
flags.DEFINE_string("decay_method", default="poly", help="Poly or cos.")
flags.DEFINE_float("min_lr_ratio", default=0.001, help="Minimum ratio learning rate.")
flags.DEFINE_float("weight_decay", default=0.001, help="Weight decay rate.")
flags.DEFINE_float("adam_epsilon", default=1e-8, help="Adam epsilon.")
flags.DEFINE_string("serving_model_dir", default="models/estimator", help="Estimator model_dir for serving service.")
# Attention 网络配置
flags.DEFINE_string("model_config_path", default="models/config.json", help="Model config path.")
flags.DEFINE_bool("use_bfloat16", False, help="Whether to use bfloat16.")
flags.DEFINE_float("dropout", default=0.1, help="Dropout rate.")
flags.DEFINE_float("dropatt", default=0.1, help="Attention dropout rate.")
flags.DEFINE_enum("init", default="normal", enum_values=["normal", "uniform"], help="Initialization method.")
flags.DEFINE_float("init_range", default=0.1, help="Initialization std when init is uniform.")
flags.DEFINE_float("init_std", default=0.02, help="Initialization std when init is normal.")
flags.DEFINE_integer("clamp_len", default=-1, help="Clamp length")
flags.DEFINE_string("summary_type", default="last", help="Method used to summarize a sequence into a compact vector.")
flags.DEFINE_bool("use_summ_proj", default=True, help="Whether to use projection for summarizing sequences.")

FLAGS = tf.app.flags.FLAGS