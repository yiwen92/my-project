import os
from absl import flags
# GPU 设置
# nvidia-smi
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

pwd_path = os.path.abspath(os.path.dirname(__file__))

TASK = 2    #   0: generate corpus, 1: preprocess, 2: pretrain, 3: finetuning run_classifier, 4: ner task, 5: race task

class Config:
    def __init__(self):
        self.original_corpus = os.path.join(pwd_path, "data/original_corpus/")
        self.pretrain_corpus = os.path.join(pwd_path, "data/pretrain_corpus/")
        self.vocab = os.path.join(pwd_path, "data/vocab")
        self.idf = os.path.join(pwd_path, "data/idf")
        self.lm = os.path.join(pwd_path, "language_model/query_word.5gram.arpa")
        self.query_len = os.path.join(pwd_path, "data/query_len")
        self.stop_words = os.path.join(pwd_path, "dict/stopwords.txt")
        self.place_names = os.path.join(pwd_path, "dict/cities.csv")
        self.rank_model = "rank_model/query_weight_xgb.model"
        self.func_file = "dict/func.txt"
        self.indus_file = "dict/ind.txt"

conf = Config()
FLAGS = flags.FLAGS
#'''
#************************************ common parameters ******************************************#
flags.DEFINE_integer("seq_len", 16, help="Sequence length.")
flags.DEFINE_integer("reuse_len", 8, help="Number of token that can be reused as memory. Could be half of `seq_len`.")
flags.DEFINE_integer("num_predict", default=8, help="Num of tokens to predict.")
flags.DEFINE_bool("do_train", default=True, help="whether to do training")
flags.DEFINE_bool("do_eval", default=True, help="whether to do eval")
flags.DEFINE_bool("do_predict", default=True, help="whether to do prediction")
flags.DEFINE_bool("use_tpu", False, help="whether to use TPUs")
flags.DEFINE_string("master", default=None, help="master")
flags.DEFINE_integer("num_hosts", default=1, help="number of TPU hosts")
flags.DEFINE_integer("max_save", default=0, help="Max number of checkpoints to save. Use 0 to save all.")
flags.DEFINE_integer("num_passes", default=1, help="Number of passed used for training.")
flags.DEFINE_integer("train_steps", default=1000000, help="Total number of training steps.")
flags.DEFINE_integer("train_batch_size", default=32, help="Size of the train batch across all hosts.")
flags.DEFINE_bool("use_bfloat16", False, help="Whether to use bfloat16.")
flags.DEFINE_float("dropout", default=0.5, help="Dropout rate.")
flags.DEFINE_float("dropatt", default=0.1, help="Attention dropout rate.")
flags.DEFINE_enum("init", default="normal", enum_values=["normal", "uniform"], help="Initialization method.")
flags.DEFINE_float("init_range", default=0.1, help="Initialization std when init is uniform.")
flags.DEFINE_float("init_std", default=0.02, help="Initialization std when init is normal.")
flags.DEFINE_integer("clamp_len", default=-1, help="Clamp length")
flags.DEFINE_integer("warmup_steps", default=1000, help="Number of steps for linear lr warmup.")
flags.DEFINE_float("learning_rate", default=1e-4, help="Maximum learning rate.")
flags.DEFINE_string("decay_method", default="poly", help="Poly or cos.")
flags.DEFINE_float("min_lr_ratio", default=0.001, help="Minimum ratio learning rate.")
flags.DEFINE_float("weight_decay", default=0.0, help="Weight decay rate.")
flags.DEFINE_float("adam_epsilon", default=1e-8, help="Adam epsilon.")
flags.DEFINE_float("clip", default=1.0, help="Gradient clipping value.")
flags.DEFINE_bool("bi_data", False, help="whether to create bidirectional data")
flags.DEFINE_integer("num_core_per_host", default=1, help="8 for TPU v2 and v3-8, 16 for larger TPU v3 pod. In the context of GPU training, it refers to the number of GPUs used.")
flags.DEFINE_integer("mask_alpha", default=6, help="How many tokens to form a group.")
flags.DEFINE_integer("mask_beta", default=1, help="How many tokens to mask within each group.")
flags.DEFINE_bool("uncased", True, help="Use uncased inputs or not.")
flags.DEFINE_integer("bsz_per_host", 32, help="batch size per host.")
flags.DEFINE_integer("iterations", default=1000, help="Number of iterations per repeat loop.")
flags.DEFINE_integer("save_steps", default=1000, help="Save the model for every save_steps. If None, not to save any model.")
flags.DEFINE_integer("max_seq_length", default=128, help="Max sequence length : (pretrain flags.seq_len)")
flags.DEFINE_bool("overwrite_data", default=False, help="If False, will use cached data if available.")
flags.DEFINE_integer("shuffle_buffer", default=2048, help="Buffer size used for shuffle.")
flags.DEFINE_string("model_config_path", default="pretrain_model/config.json", help="Model config path.")
flags.DEFINE_string("summary_type", default="last", help="Method used to summarize a sequence into a compact vector.")
flags.DEFINE_bool("use_summ_proj", default=True, help="Whether to use projection for summarizing sequences.")
if TASK == 1:
    #************************************ preprocess data_utils.py parameters ******************************************#
    flags.DEFINE_bool("use_eod", True, help="whether to append EOD at the end of a doc.")
    flags.DEFINE_bool("from_raw_text", True, help="Whether the input is raw text or encoded ids.")
    flags.DEFINE_string("input_glob", "data/pretrain_corpus/*.txt", help="Input file glob.")
    flags.DEFINE_string("sp_path", "token_model/chinese/spiece.model", help="Path to the sentence piece model.")
    flags.DEFINE_string("save_dir", "proc_data/pretrain_corpus", help="Directory for saving the processed data.")
    flags.DEFINE_enum("split", "train", ["train", "dev", "test"], help="Save the data as which split.")
    flags.DEFINE_integer("pass_id", 0, help="ID of the current pass. Different passes sample different negative segment.")
    flags.DEFINE_integer("num_task", 1, help="Number of total tasks.")
    flags.DEFINE_integer("task", 0, help="The Task ID. This value is used when using multiple workers to identify each worker.")
elif TASK == 2:
    #************************************ pretrain train.py parameters ******************************************#
    flags.DEFINE_integer("perm_size", 8, help="Window size of permutation.")
    flags.DEFINE_integer("n_token", 32000, help="Vocab size")
    flags.DEFINE_string("model_dir", default="pretrain_model", help="Estimator model_dir.")
    flags.DEFINE_string("record_info_dir", default="proc_data/pretrain_corpus/tfrecords", help="Path to local directory containing `record_info-lm.json`.")
    flags.DEFINE_integer("mem_len", default=0, help="Number of steps to cache")
    flags.DEFINE_integer("batch_size", default=32, help="batch size of every train step")
    flags.DEFINE_integer("n_layer", default=6, help="Number of layers.")
    flags.DEFINE_integer("d_model", default=100, help="Dimension of the model. embedding of input token")
    flags.DEFINE_integer("n_head", default=1, help="Number of attention heads.")
    flags.DEFINE_integer("d_head", default=32, help="Dimension of each attention head.")
    flags.DEFINE_integer("d_inner", default=128, help="Dimension of inner hidden size in positionwise feed-forward.")
    flags.DEFINE_string("ff_activation", default="relu", help="Activation type used in position-wise feed-forward.")
    flags.DEFINE_bool("untie_r", default=True, help="Untie r_w_bias and r_r_bias")
    flags.DEFINE_bool("same_length", default=False, help="Same length attention")
    flags.DEFINE_string("init_checkpoint", default="pretrain_model", help="Checkpoint path for initializing the model.")   # "pretrain_model"
elif TASK == 3:
    #************************************ finetuning run_classifier.py parameters ******************************************#
    flags.DEFINE_string("output_dir", default="proc_data/imdb", help="Output dir for TF records.")
    flags.DEFINE_string("task_name", default="imdb", help="Task name")
    flags.DEFINE_bool("is_regression", default=False, help="Whether it's a regression task.")
    flags.DEFINE_string("spiece_model_file", default="token_model/english/spiece.model", help="Sentence Piece model path.")
    flags.DEFINE_string("data_dir", default="data/aclImdb", help="Directory for input data.")
    flags.DEFINE_string("cls_scope", default=None, help="Classifier layer scope.")
    flags.DEFINE_string("model_dir", default="finetuning_model/imdb", help="Directory for saving the finetuned model.")
    flags.DEFINE_string("init_checkpoint", default="pretrain_model/model.ckpt-35", help="checkpoint path for initializing the model. Could be a pretrained model or a finetuned model.")
elif TASK == 4:
    flags.DEFINE_string("output_dir", default="proc_data/ner", help="Output dir for TF records.")
    flags.DEFINE_string("task_name", default="ner", help="Task name")
    flags.DEFINE_string("spiece_model_file", default="token_model/chinese/spiece.model", help="Sentence Piece model path.")
    flags.DEFINE_string("model_dir", default="finetuning_model/ner", help="Directory for saving the finetuned model.")
    flags.DEFINE_string("data_dir", default="data/ner", help="Directory for input data.")
    flags.DEFINE_string("init_checkpoint", default="pretrain_model/model.ckpt-35", help="checkpoint path for initializing the model. Could be a pretrained model or a finetuned model.")
    flags.DEFINE_string("eval_split", default="dev", help="could be dev or test")
    flags.DEFINE_integer("eval_batch_size", default=8, help="batch size for evaluation")
    flags.DEFINE_bool("eval_all_ckpt", default=False, help="Eval all ckpts. If False, only evaluate the last one.")
    flags.DEFINE_string("predict_dir", default="predict/ner", help="Dir for saving prediction files.")
    flags.DEFINE_string("predict_ckpt", default=None, help="Ckpt path for do_predict. If None, use the last one.")
elif TASK == 5:
    flags.DEFINE_string("init_checkpoint", default="pretrain_model/model.ckpt-35", help="checkpoint path for initializing the model. Could be a pretrained model or a finetuned model.")
    flags.DEFINE_string("output_dir", default="./proc_data/race", help="Output dir for TF records.")
    flags.DEFINE_string("spiece_model_file", default="token_model/english/spiece.model", help="Sentence Piece model path.")
    flags.DEFINE_string("model_dir", default="finetuning_model/race", help="Directory for saving the finetuned model.")
    flags.DEFINE_string("data_dir", default="./data/race", help="Directory for input data.")
    flags.DEFINE_bool("high_only", default=False, help="Evaluate on high school only.")
    flags.DEFINE_bool("middle_only", default=False, help="Evaluate on middle school only.")
    flags.DEFINE_integer("max_qa_length", default=128, help="Max length for the concatenated question and answer.")
    pass
#'''
