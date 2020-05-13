import random, json, collections, os, re, copy
import numpy as np
from tqdm import tqdm
from utils import clean_line, re_en, token2list
from collections import defaultdict
from config import MAX_NUM_NEG, conf, VOCAB_SIZE, SEQ_LEN, FLAGS
from seg_utils import Tokenizer, is_valid_tokens
import tensorflow as tf

word2id = json.load(open(conf.vocab, encoding="utf8"))

class TrainData():
    def __init__(self):
        self.tokenizer = Tokenizer()

    def original2corp(self):
        text =[]
        print("extract corpu from original file: %s --> corpus file: %s" % (FLAGS.original_file, FLAGS.corpus_file))
        for line in open(FLAGS.original_file, encoding="utf8").readlines():
            try: e = line.strip().split("\t")[33].replace("\\n", "").lower()
            except: continue
            a=line.strip().split("\t")
            text.append(e)
        with open(FLAGS.corpus_file, "w", encoding="utf8") as fin:
            fin.write("\n".join(text))

    def gen_train_samples(self):
        self.original2corp()
        sample_set = {}
        np.random.seed(8)
        # 加载数据，以文本为单位
        important_tokens = []
        text = open(FLAGS.corpus_file, encoding="utf8").readlines()[:10]
        print("select important tokens...")
        for e in tqdm(text, total=len(text)):
            tmp = self.tokenizer.select_important_tokens(clean_line(e.strip()))
            if len(tmp) < 10: continue
            important_tokens.append(tmp)
        # 采样正负样本，同一个文本中的词为正样本，不同文本中的词为负样本
        print("sample(1+k negative) train and valid set...")
        num_neg = min(len(important_tokens) - 1, MAX_NUM_NEG)
        for cur_index, cur_ele in tqdm(enumerate(important_tokens), total=len(important_tokens)):
            np.random.shuffle(cur_ele)
            cut_index = int(len(cur_ele) / 3)
            lhs, rhs = cur_ele[: cut_index], cur_ele[cut_index:]
            for word_index, word in enumerate(lhs):
                if word in sample_set: continue
                positive_entity = rhs[word_index]  # 正样本
                # 负采样
                negative_entitys, negs = [], []
                negative_indexes = [i for i in range(len(important_tokens)) if i != cur_index]
                random.shuffle(negative_indexes)
                for e in negative_indexes:
                    if (len(negs) >= num_neg): break
                    if word in important_tokens[e] or positive_entity in important_tokens[e]: continue
                    negs.append(e)
                for neg_index in negs:
                    while True:
                        neg_tmp = random.sample(important_tokens[neg_index], 1)[0]
                        if neg_tmp != word and neg_tmp not in negative_entitys: break
                    negative_entitys.append(neg_tmp)
                assert len(negative_entitys) == num_neg
                # 采样数少的情况下进行填充
                #if len(negative_entitys) < num_neg:
                #    negative_entitys += ["PAD"] * (num_neg - len(negative_entitys))
                sample_set[word] = [positive_entity, negative_entitys]
        # 产生字典
        token_freq = defaultdict(int); token_freq["UNKNOWN"] = 1e8; #token_freq["PAD"] = 1e8-1
        for k, (p, n) in sample_set.items():
            tmp = [k, p] + n
            for t in tmp:
                if re_en.fullmatch(t): token_freq[t] += 1
                else:
                    for e in list(t):
                        token_freq[e] += 1
        sorted_token_freq = sorted(token_freq.items(), key=lambda d: d[1], reverse=True)[:VOCAB_SIZE]
        word2id = {w: i for i, (w, f) in enumerate(sorted_token_freq)}
        if conf.over_write_vocab:
            print("generate word2id file: %s" % (conf.vocab))
            json.dump(word2id, open(conf.vocab, "w", encoding="utf8"), ensure_ascii=False, indent=2)
        _keys_ = list(sample_set.keys())
        train_set = {k: sample_set[k] for k in _keys_[:int(len(_keys_) * conf.train_valid_ratio)]}
        valid_set = {k: sample_set[k] for k in _keys_[int(len(_keys_) * conf.train_valid_ratio):]}
        print("total_sample: %d\ttrain_sample: %d\tvalid_sample :%d" % (len(sample_set), len(train_set), len(valid_set)))
        print("generate train sample file :%s\tvalid sample file: %s" % (conf.train_samples, conf.valid_samples))
        json.dump(train_set, open(conf.train_samples, "w", encoding="utf8"), ensure_ascii=False, indent=2)
        json.dump(valid_set, open(conf.valid_samples, "w", encoding="utf8"), ensure_ascii=False, indent=2)

    def gen_vocab(self, title2entitys):
        token_freq = defaultdict(int); token_freq["UNKNOWN"] = 1e8
        for title, entitys in title2entitys.items():
          line = [title] + entitys
          for t in line:
              if re_en.fullmatch(t): token_freq[t] += 1
              else:
                  for e in list(t): token_freq[e] += 1
        sorted_token_freq = sorted(token_freq.items(), key=lambda d: d[1], reverse=True)[:VOCAB_SIZE]
        word2id = {w: i for i, (w, f) in enumerate(sorted_token_freq)}
        print("generate word2id file: %s" % (conf.vocab))
        json.dump(word2id, open(conf.vocab, "w", encoding="utf8"), ensure_ascii=False, indent=2)

    def gen_train_sample_based_title_desc(self):
        entity_dicts = {line.strip(): 1 for line in open(conf.new_entity_file, encoding="utf8").readlines()}
        valid_titles = {line.strip(): 1 for line in open("data/valid_titles", encoding="utf8").readlines()}
        title_entitys, entity_title, sample_set = {}, {}, []
        matchObj = re.compile(r'(.+)&([0-9]+)', re.M | re.I)
        title2entitys = {line.strip().lower().split('\t')[0]: line.strip().lower().split('\t')[1:] \
          for line in open("data/cv_title2entitys_corpu", encoding="utf8").readlines()}
        title_entitys = {k: v for k, v in title2entitys.items() if len(v) >= 10 and len(v) < 20}
        if conf.over_write_vocab: self.gen_vocab(title_entitys)
        _keys_ = list(title_entitys.keys())
        print("sample(1+k negative) train and valid set...")
        num_neg = min(len(title_entitys) - 1, MAX_NUM_NEG)
        # 采样
        for title, entitys in tqdm(title_entitys.items(), total=len(title_entitys)):
            positive_entitys = random.sample(entitys, min(len(entitys), 10))         # 正样本
            negative_titles_candidate = [e for e in _keys_ if e != title]
            for pos_entity in positive_entitys:     # 负样本
              negative_entitys = []
              negs = random.sample(negative_titles_candidate, num_neg)
              for neg_tit in negs:
                try: negative_entitys.append(random.sample(title_entitys[neg_tit], 1)[0])
                except:
                  a=1
              if len(negative_entitys) < num_neg:
                negative_entitys += [negative_entitys[0]] * (num_neg - len(negative_entitys))
              assert len(negative_entitys) == num_neg
              sample_set.append([title, pos_entity, list(negative_entitys)])
        #exit()
        train_set = {i: ele for i, ele in enumerate(sample_set[: int(len(sample_set) * conf.train_valid_ratio)])}
        valid_set = {i: ele for i, ele in enumerate(sample_set[int(len(sample_set) * conf.train_valid_ratio): ])}
        print("total_sample: %d\ttrain_sample: %d\tvalid_sample :%d" % (len(sample_set), len(train_set), len(valid_set)))
        print("generate train sample file :%s\tvalid sample file: %s" % (FLAGS.train_samples, FLAGS.valid_samples))
        json.dump(train_set, open(FLAGS.train_samples, "w", encoding="utf8"), ensure_ascii=False, indent=2)
        json.dump(valid_set, open(FLAGS.valid_samples, "w", encoding="utf8"), ensure_ascii=False, indent=2)
        a=1

def batch_iter(x, y, label, batch_size=64):
    np.random.seed(8)
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]
    label_shuffle = label[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id: end_id], y_shuffle[start_id: end_id], label_shuffle[start_id: end_id]

def seq2ids(text, max_length=SEQ_LEN):
    """将文本转换为id表示"""
    data = token2list(text)
    data_id = [word2id.get(x, 0) for x in data]
    x_pad = data_id
    if len(data_id) > max_length: x_pad = data_id[: max_length]
    else: x_pad = data_id + [0] * (max_length - len(data_id))
    return x_pad

class train_sample():
    def __init__(self, entity, pos_entity, neg_entitys):
        self.entity = entity
        self.pos_entity = pos_entity
        self.neg_entitys = neg_entitys

def gen_train_samples(file_path):
    train_samples = json.load(open(file_path, encoding="utf8"))
    samples = []
    #for k, (p, n) in train_samples.items():
    for i, (k, p, n) in train_samples.items():
        kid = seq2ids(k)
        pid = seq2ids(p)
        nid = [seq2ids(e) for e in n]
        ts = train_sample(kid, pid, nid)
        samples.append(ts)
    X = np.array([e.entity for e in samples])           #  单个实体
    Y = np.array([[e.pos_entity] + e.neg_entitys for e in samples])         # 正样本 + k个负样本
    label = np.array([[1] + [0] * (Y.shape[1] - 1)] * Y.shape[0])  # 1 + k 个标签，第一个为1，其它的为0
    return X, Y, label

#************************************************ tensorflow tf.record 训练数据构建 ************************************************#
class InputExample(object):
  """A single training/test example."""
  def __init__(self, guid, entity, pos_entity, neg_entitys):
      self.guid = guid
      self.entity = entity
      self.pos_entity = pos_entity
      self.neg_entitys = neg_entitys

class InputFeatures(object):
  """A single set of features of data."""
  def __init__(self, entity_ids, pos_entity_id, neg_entity_list_ids, labels):
    self.entity_ids = entity_ids
    self.pos_entity_id = pos_entity_id
    self.neg_entity_list_ids = neg_entity_list_ids
    self.labels = labels

def create_example(file_path):
    examples = []
    train_samples = json.load(open(file_path, encoding="utf8"))
    for i, (ent, pos, negs) in train_samples.items():
    #for i, (ent, (pos, negs)) in enumerate(train_samples.items()):
        examples.append(InputExample(i, ent, pos, negs))
    return examples

def file_based_convert_examples_to_features(examples, max_seq_length, tokenize_fn, output_file):
  """Convert a set of `InputExample`s to a TFRecord file."""
  # do not create duplicated records
  if tf.gfile.Exists(output_file) and not FLAGS.overwrite_data:
    tf.logging.info("Do not overwrite tfrecord {} exists.".format(output_file))
    return

  tf.logging.info("Create new tfrecord {}.".format(output_file))

  writer = tf.python_io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example {} of {}".format(ex_index, len(examples)))

    # 转变单个样本到 tf.record 数据
    feature = convert_single_example(ex_index, example, max_seq_length, tokenize_fn)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    def create_int_feature_list(values):
        d0 = [[1, 2], [3, 4]]
        d = np.array(d0).astype(np.int8).tobytes()
        a = np.array(values).tolist()
        values = np.array(values).tobytes()
        f = tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))
        return f

    def get_tfrecords_example(feature):
        tfrecords_features = collections.OrderedDict()
        tfrecords_features['entity_ids'] = tf.train.Feature(int64_list=tf.train.Int64List(value=list(feature.entity_ids)))
        tfrecords_features['labels'] = tf.train.Feature(int64_list=tf.train.Int64List(value=list(feature.labels)))
        entitys = copy.deepcopy(feature.pos_entity_id)
        for e in feature.neg_entity_list_ids: entitys.extend(e)
        feat_shape = [1 + len(feature.neg_entity_list_ids), len(feature.neg_entity_list_ids[0])]
        assert feat_shape[0] * feat_shape[1] == len(entitys)
        tfrecords_features['entity_ids_list'] = tf.train.Feature(int64_list=tf.train.Int64List(value=list(entitys)))
        tfrecords_features['shape'] = tf.train.Feature(int64_list=tf.train.Int64List(value=list(feat_shape)))
        return tf.train.Example(features=tf.train.Features(feature=tfrecords_features))
    '''
    features = collections.OrderedDict()
    features["entity_ids"] = create_int_feature(feature.entity_ids)
    for i, e in enumerate([feature.pos_entity_id] + feature.neg_entity_list_ids):
        a=1
    features["entity_ids_list"] = create_int_feature_list([feature.pos_entity_id] + feature.neg_entity_list_ids)
    #features["entity_ids_list"] = [create_int_feature(e) for e in [feature.pos_entity_id] + feature.neg_entity_list_ids]
    #features["entity_ids_list"] = create_int_feature_lists([feature.pos_entity_id] + feature.neg_entity_list_ids)

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    '''
    tf_example = get_tfrecords_example(feature=feature)
    writer.write(tf_example.SerializeToString())
  writer.close()

def convert_single_example(ex_index, example, max_seq_length, tokenize_fn):
  """Converts a single `InputExample` into a single `InputFeatures`."""
  ent_ids = tokenize_fn(example.entity)
  pos_ent_ids = tokenize_fn(example.pos_entity)
  neg_entity_list_ids = [tokenize_fn(e) for e in example.neg_entitys]
  labels = [1] + [0] * len(neg_entity_list_ids)  # 1 + k 个标签，第一个为1，其它的为0

  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("entity_ids: %s" % " ".join([str(x) for x in ent_ids]))
    tf.logging.info("positive_entity_ids: %s" % " ".join([str(x) for x in pos_ent_ids]))
    tf.logging.info("negative_entity_list_ids: %s" % "\t".join([",".join([str(x) for x in e]) for e in neg_entity_list_ids]))
    tf.logging.info("labels: %s" % " ".join([str(x) for x in labels]))

  feature = InputFeatures(entity_ids=ent_ids, pos_entity_id=pos_ent_ids, neg_entity_list_ids=neg_entity_list_ids, labels=labels)
  return feature

def tokenize_fn(text):
    return seq2ids(text)

def file_based_input_fn_builder(input_file, entity_num, seq_length, is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""
  # 一个训练样本的配置
  name_to_features = {
      "entity_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "entity_ids_list": tf.FixedLenFeature([entity_num * seq_length], tf.int64),
      "labels": tf.FixedLenFeature([entity_num], tf.int64),
  }

  tf.logging.info("Input tfrecord file {}".format(input_file))

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)
    '''
    entity_ids_list = example.pop("entity_ids_list")
    entity_ids_list = tf.decode_raw(entity_ids_list, tf.int64)
    example['entity_ids_list'] = tf.reshape(entity_ids_list, [entity_num, seq_length])
    '''
    #entity_ids_list = example['entity_ids_list']
    #example['entity_ids_list'] = tf.reshape(entity_ids_list, [example['shape'][0], example['shape'][1]])

    return example

  def input_fn(params, input_context=None):
    """The actual input function."""
    if FLAGS.use_tpu:
      batch_size = params["batch_size"]
    elif is_training:
      batch_size = FLAGS.train_batch_size
    elif FLAGS.do_eval:
      batch_size = FLAGS.eval_batch_size
    else:
      batch_size = FLAGS.predict_batch_size

    d = tf.data.TFRecordDataset(input_file)
    # Shard the dataset to difference devices
    if input_context is not None:
      tf.logging.info("Input pipeline id %d out of %d",
          input_context.input_pipeline_id, input_context.num_replicas_in_sync)
      d = d.shard(input_context.num_input_pipelines,
                  input_context.input_pipeline_id)

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    if is_training:
      d = d.shuffle(buffer_size=FLAGS.shuffle_buffer)
      d = d.repeat()

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn

def gen_train_input_fn(file_path):
    #tf.logging.set_verbosity(tf.logging.INFO)
    if not tf.gfile.Exists(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)
    train_file_base = "{}.len-{}.train.tf_record".format("entity", FLAGS.max_seq_length)
    train_file = os.path.join(FLAGS.output_dir, train_file_base)
    tf.logging.info("Use tfrecord file {}".format(train_file))
    train_examples = create_example(file_path)
    np.random.shuffle(train_examples)
    tf.logging.info("Num of train samples: {}".format(len(train_examples)))
    file_based_convert_examples_to_features(train_examples, FLAGS.max_seq_length, tokenize_fn, train_file)
    neg_num = len(train_examples[0].neg_entitys)
    train_input_fn = file_based_input_fn_builder(input_file=train_file, entity_num=1+neg_num, seq_length=FLAGS.max_seq_length, is_training=True, drop_remainder=True)
    return train_input_fn

if __name__ == "__main__":
    gen_train_input_fn(FLAGS.train_samples); exit()
    td = TrainData()
    #td.gen_train_samples()
    #td.gen_train_sample_based_title_desc()
    #gen_train_samples(FLAGS.valid_samples)