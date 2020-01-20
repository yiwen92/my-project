from __future__ import absolute_import, division, print_function

from os.path import join
import os, collections, json, codecs, regex
import numpy as np
import tensorflow as tf
import sentencepiece as spm
from prepro_utils import preprocess_text, encode_ids
from absl import flags
import model_utils
import seq2seq_models

TaskType = 'de-en' #"nlpcc2018+hsk" #"de-en"
SEG_ID_A   = 0
SEG_ID_PAD = 4
#'''
# Model
flags.DEFINE_string("model_config_path", default="model/config.json", help="Model config path.")
flags.DEFINE_float("dropout", default=0.1, help="Dropout rate.")
flags.DEFINE_float("dropatt", default=0.1, help="Attention dropout rate.")
flags.DEFINE_integer("clamp_len", default=-1, help="Clamp length")
flags.DEFINE_bool("use_bfloat16", False, help="Whether to use bfloat16.")

# Parameter initialization
flags.DEFINE_enum("init", default="normal", enum_values=["normal", "uniform"], help="Initialization method.")
flags.DEFINE_float("init_std", default=0.02, help="Initialization std when init is normal.")
flags.DEFINE_float("init_range", default=0.1, help="Initialization std when init is uniform.")

# I/O paths
flags.DEFINE_bool("overwrite_data", default=False, help="If False, will use cached data if available.")
flags.DEFINE_string("init_checkpoint", default="model/" + TaskType + "/model.ckpt-0",
      help="checkpoint path for initializing the model. "
      "Could be a pretrained model or a finetuned model.")
flags.DEFINE_string("output_dir", default="proc_data/" + TaskType, help="Output dir for TF records.")
flags.DEFINE_string("spiece_model_file", default="token_model/english/spiece.model", help="Sentence Piece model path.")
flags.DEFINE_string("model_dir", default="model/" + TaskType, help="Directory for saving the finetuned model.")
flags.DEFINE_string("data_dir", default="data/" + TaskType, help="Directory for input data.")

# TPUs and machines
flags.DEFINE_bool("use_tpu", default=False, help="whether to use TPU.")
flags.DEFINE_integer("num_hosts", default=1, help="How many TPU hosts.")
flags.DEFINE_integer("num_core_per_host", default=1, help="8 for TPU v2 and v3-8, 16 for larger TPU v3 pod. In the context "
      "of GPU training, it refers to the number of GPUs used.")
flags.DEFINE_string("tpu_job_name", default=None, help="TPU worker job name.")
flags.DEFINE_string("tpu", default=None, help="TPU name.")
flags.DEFINE_string("tpu_zone", default=None, help="TPU zone.")
flags.DEFINE_string("gcp_project", default=None, help="gcp project.")
flags.DEFINE_string("master", default=None, help="master")
flags.DEFINE_integer("iterations", default=1000, help="number of iterations per TPU training loop.")

# training
flags.DEFINE_bool("do_train", default=True, help="whether to do training")
flags.DEFINE_integer("train_steps", default=10000, help="Number of training steps")
flags.DEFINE_integer("warmup_steps", default=500, help="number of warmup steps")
flags.DEFINE_float("learning_rate", default=2e-5, help="initial learning rate")
flags.DEFINE_float("lr_layer_decay_rate", 1.0, "Top layer: lr[L] = FLAGS.learning_rate. Low layer: lr[l-1] = lr[l] * lr_layer_decay_rate.")
flags.DEFINE_float("min_lr_ratio", default=0.0, help="min lr ratio for cos decay.")
flags.DEFINE_float("clip", default=1.0, help="Gradient clipping")
flags.DEFINE_integer("max_save", default=0, help="Max number of checkpoints to save. Use 0 to save all.")
flags.DEFINE_integer("save_steps", default=500, help="Save the model for every save_steps. If None, not to save any model.")
flags.DEFINE_integer("train_batch_size", default=32, help="Batch size for training")
flags.DEFINE_float("weight_decay", default=0.00, help="Weight decay rate")
flags.DEFINE_float("adam_epsilon", default=1e-8, help="Adam epsilon")
flags.DEFINE_string("decay_method", default="poly", help="poly or cos")

# evaluation
flags.DEFINE_bool("do_eval", default=False, help="whether to do eval")
flags.DEFINE_bool("do_predict", default=True, help="whether to do prediction")
flags.DEFINE_string("eval_split", default="dev", help="could be dev or test")
flags.DEFINE_integer("eval_batch_size", default=8, help="batch size for evaluation")
flags.DEFINE_integer("predict_batch_size", default=128, help="batch size for prediction.")
flags.DEFINE_string("predict_dir", default="predict/" + TaskType, help="Dir for saving prediction files.")
flags.DEFINE_bool("eval_all_ckpt", default=False, help="Eval all ckpts. If False, only evaluate the last one.")
flags.DEFINE_string("predict_ckpt", default=None, help="Ckpt path for do_predict. If None, use the last one.")

# task specific
flags.DEFINE_integer("max_seq_length", default=128, help="Max sequence length")
flags.DEFINE_integer("shuffle_buffer", default=2048, help="Buffer size used for shuffle.")
flags.DEFINE_integer("num_passes", default=1, help="Num passes for processing training data. This is use to batch data without loss for TPUs.")
flags.DEFINE_bool("uncased", default=False, help="Use uncased.")
flags.DEFINE_string("cls_scope", default=None, help="Classifier layer scope.")

# extra parameters
flags.DEFINE_integer("min_cnt", default=20, help="word counts whose occurred less than min_cnt are encoded as <UNK>.")
flags.DEFINE_integer("n_layer", default=1, help="layers of transformer.")
flags.DEFINE_integer("d_model", default=100, help="embedding of tokens.")
flags.DEFINE_integer("n_head", default=4, help="head number of attention.")
flags.DEFINE_integer("d_head", default=8, help="dimension of attention head.")
flags.DEFINE_integer("d_inner", default=128, help="dimension of model inner.")
flags.DEFINE_string("ff_activation", default="relu", help="activation function of neural network.")
flags.DEFINE_bool("untie_r", default=True, help="whether to use bias.")

FLAGS = flags.FLAGS
#'''

class InputFeatures(object):
  """A single set of features of data."""
  def __init__(self, input_ids, input_mask, input_segment_ids, output_ids, output_mask, output_segment_ids):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.input_segment_ids = input_segment_ids
    self.output_ids = output_ids
    self.output_mask = output_mask
    self.output_segment_ids = output_segment_ids

class InputExample(object):
  def __init__(self, guid, source_ids, target_ids):
    self.guid = guid
    self.source_ids = source_ids
    self.target_ids = target_ids

class Seq2SeqProcessor(object):
    def __init__(self):
        self.vocab_src = None
        self.vocab_trg = None

class De2EnTranslateProcessor(Seq2SeqProcessor):
  def __init__(self):
      super(De2EnTranslateProcessor, self).__init__()
      self.train_src_file = FLAGS.data_dir + "/train.tags.de-en1.de"
      self.train_trg_file = FLAGS.data_dir + "/train.tags.de-en1.en"
      self.test_src_file = FLAGS.data_dir + "/IWSLT16.TED.tst2014.de-en.de.xml"
      self.test_trg_file = FLAGS.data_dir + "/IWSLT16.TED.tst2014.de-en.en.xml"
      self.vocab_src_file = FLAGS.data_dir + "/de.vocab.tsv"
      self.vocab_trg_file = FLAGS.data_dir + "/en.vocab.tsv"
      self.load_src_vocab()
      self.load_trg_vocab()

  def load_src_vocab(self):
    src_vocab = [line.split()[0] for line in codecs.open(self.vocab_src_file, 'r', 'utf-8').read().splitlines() if int(line.split()[1]) >= FLAGS.min_cnt]
    self.src_word2idx = {word: idx for idx, word in enumerate(src_vocab)}
    self.src_idx2word = {idx: word for idx, word in enumerate(src_vocab)}

  def load_trg_vocab(self):
      trg_vocab = [line.split()[0] for line in codecs.open(self.vocab_trg_file, 'r', 'utf-8').read().splitlines() if int(line.split()[1]) >= FLAGS.min_cnt]
      self.trg_word2idx = {word: idx for idx, word in enumerate(trg_vocab)}
      self.trg_idx2word = {idx: word for idx, word in enumerate(trg_vocab)}

  def get_train_examples(self):
      examples = []
      def _refine(line):
          line = regex.sub("[^\s\p{Latin}']", "", line)
          return line.strip()
      source_sents = [_refine(line) for line in codecs.open(self.train_src_file, 'r', 'utf-8').read().split('\n') if line and line[0] != "<"]
      target_sents = [_refine(line) for line in codecs.open(self.train_trg_file, 'r', 'utf-8').read().split('\n') if line and line[0] != '<']
      for source_sent, target_sent in zip(source_sents, target_sents):
          source_ids = [self.src_word2idx.get(word, 1) for word in (source_sent + u" </S>").split()]  # 1: OOV, </S>: End of Text
          target_ids = [self.trg_word2idx.get(word, 1) for word in (target_sent + u" </S>").split()]
          examples.append(InputExample(guid="unused_id", source_ids=source_ids, target_ids=target_ids))
      return examples

  def get_dev_examples(self):
      examples = []
      def _refine(line):
          line = regex.sub("<[^>]+>", "", line)
          line = regex.sub("[^\s\p{Latin}']", "", line)
          return line.strip()
      source_sents = [_refine(line) for line in codecs.open(self.test_src_file, 'r', 'utf-8').read().split('\n') if line and line[:4] == "<seg"]
      target_sents = [_refine(line) for line in codecs.open(self.test_trg_file, 'r', 'utf-8').read().split('\n') if line and line[:4] == '<seg']
      for source_sent, target_sent in zip(source_sents, target_sents):
          source_ids = [self.src_word2idx.get(word, 1) for word in (source_sent + u" </S>").split()]  # 1: OOV, </S>: End of Text
          target_ids = [self.trg_word2idx.get(word, 1) for word in (target_sent + u" </S>").split()]
          examples.append(InputExample(guid="unused_id", source_ids=source_ids, target_ids=target_ids))
      return examples

class CorrectProcessor(Seq2SeqProcessor):
    def __init__(self):
        super(CorrectProcessor, self).__init__()
        pass

def convert_single_example(ex_index, example, max_seq_length):
  """Converts a single `InputExample` into a single `InputFeatures`."""
  input_ids = example.source_ids
  output_ids = example.target_ids
  # Modifies `tokens_a` and `tokens_b` in place so that the total length is less than the specified length.
  if len(input_ids) > max_seq_length:
      input_ids = input_ids[:max_seq_length]
  if len(output_ids) > max_seq_length:
      output_ids = output_ids[:max_seq_length]

  input_segment_ids = [SEG_ID_A] * len(input_ids)
  output_segment_ids = [SEG_ID_A] * len(output_ids)

  # The mask has 0 for real tokens and 1 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [0] * len(input_ids)
  output_mask = [0] * len(output_ids)

  # Zero-pad up to the sequence length.
  if len(input_ids) < max_seq_length:
    delta_len = max_seq_length - len(input_ids)
    input_ids = [0] * delta_len + input_ids
    input_mask = [1] * delta_len + input_mask
    input_segment_ids = [SEG_ID_PAD] * delta_len + input_segment_ids
  if len(output_ids) < max_seq_length:
    delta_len = max_seq_length - len(output_ids)
    output_ids = [0] * delta_len + output_ids
    output_mask = [1] * delta_len + output_mask
    output_segment_ids = [SEG_ID_PAD] * delta_len + output_segment_ids

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(input_segment_ids) == max_seq_length
  assert len(output_ids) == max_seq_length
  assert len(output_mask) == max_seq_length
  assert len(output_segment_ids) == max_seq_length

  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("input_segment_ids: %s" % " ".join([str(x) for x in input_segment_ids]))
    tf.logging.info("output_ids: %s" % " ".join([str(x) for x in output_ids]))
    tf.logging.info("output_mask: %s" % " ".join([str(x) for x in output_mask]))
    tf.logging.info("output_segment_ids: %s" % " ".join([str(x) for x in output_segment_ids]))

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      input_segment_ids=input_segment_ids,
      output_ids=output_ids,
      output_mask=output_mask,
      output_segment_ids=output_segment_ids)
  return feature

def file_based_convert_examples_to_features(examples, max_seq_length, output_file, num_passes=1):
  """Convert a set of `InputExample`s to a TFRecord file."""

  # do not create duplicated records
  if tf.gfile.Exists(output_file) and not FLAGS.overwrite_data:
    tf.logging.info("Do not overwrite tfrecord {} exists.".format(output_file))
    return

  tf.logging.info("Create new tfrecord {}.".format(output_file))

  writer = tf.python_io.TFRecordWriter(output_file)

  if num_passes > 1:
    examples *= num_passes

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example {} of {}".format(ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, max_seq_length)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    def create_float_feature(values):
      f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
      return f

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_float_feature(feature.input_mask)
    features["input_segment_ids"] = create_int_feature(feature.input_segment_ids)
    features["output_ids"] = create_int_feature(feature.output_ids)
    features["output_mask"] = create_float_feature(feature.output_mask)
    features["output_segment_ids"] = create_int_feature(feature.output_segment_ids)

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()

def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.float32),
      "output_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "output_mask": tf.FixedLenFeature([seq_length], tf.float32)
  }

  tf.logging.info("Input tfrecord file {}".format(input_file))

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.cast(t, tf.int32)
      example[name] = t

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

def get_model_fn():
  def model_fn(features, labels, mode, params):
    #### Training or Evaluation
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    #### Get loss from inputs
    inp_ids = tf.transpose(features["input_ids"], [1, 0])
    inp_mask = tf.transpose(features["input_mask"], [1, 0])
    output_ids = features["output_ids"]
    output_mask = tf.transpose(features["output_mask"], [1, 0])
    # define decoder inputs
    decoder_inputs = tf.concat((tf.ones_like(output_ids[:, :1]) * 2, output_ids[:, :-1]), -1)  # 2代表<S>，是decoder的初始输入
    decoder_inputs = tf.transpose(decoder_inputs, [1, 0])
    args = dict(
        FLAGS=FLAGS,
        is_training=is_training,
        inp_ids=inp_ids,
        inp_mask=inp_mask,
        source_ntoken=params.get("source_ntoken"),
        target_ntoken=params.get("target_ntoken"),
        output_mask=output_mask,
        output_ids=output_ids,
        decoder_inputs=decoder_inputs)
    s2sm = seq2seq_models.seq2seqmodel(**args)
    total_loss, per_example_loss, logits = s2sm.total_loss, s2sm.per_example_loss, s2sm.logits

    #### Check model parameters
    num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
    tf.logging.info('#params: {}'.format(num_params))

    #### load pretrained models
    scaffold_fn = model_utils.init_from_checkpoint(FLAGS)

    #### Evaluation mode
    if mode == tf.estimator.ModeKeys.EVAL:
      assert FLAGS.num_hosts == 1

      def metric_fn(per_example_loss, logits, output_ids):
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        istarget = tf.to_float(tf.not_equal(output_ids, 0))
        accuracy = tf.reduce_sum(tf.to_float(tf.equal(predictions, output_ids)) * istarget / (tf.reduce_sum(istarget)))
        loss = tf.metrics.mean(values=per_example_loss)
        return {
            'eval_accuracy': accuracy,
            'eval_loss': loss,
        }

      #### Constucting evaluation TPUEstimatorSpec with new cache.
      metric_args = [per_example_loss, logits, output_ids]
      eval_spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=total_loss,
            export_outputs=metric_fn(*metric_args))

      return eval_spec

    elif mode == tf.estimator.ModeKeys.PREDICT:
      pred = tf.argmax(logits, axis=-1, output_type=tf.int32)
      predictions = {
          "logits": logits,
          "pred": pred,
          "output_ids":output_ids,
      }
      output_spec = tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
      return output_spec

    #### Configuring the optimizer
    train_op, learning_rate, _ = model_utils.get_train_op(FLAGS, total_loss)

    monitor_dict = {}
    monitor_dict["lr"] = learning_rate

    #### Constucting training TPUEstimatorSpec with new cache.
    train_spec = tf.estimator.EstimatorSpec(mode=mode, loss=total_loss, train_op=train_op)

    return train_spec

  return model_fn

def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  #### Validate flags
  if FLAGS.save_steps is not None:
    FLAGS.iterations = min(FLAGS.iterations, FLAGS.save_steps)

  if FLAGS.do_predict:
    predict_dir = FLAGS.predict_dir
    if not tf.gfile.Exists(predict_dir):
      tf.gfile.MakeDirs(predict_dir)

  if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
    raise ValueError(
        "At least one of `do_train`, `do_eval, `do_predict` or "
        "`do_submit` must be True.")

  if not tf.gfile.Exists(FLAGS.output_dir):
    tf.gfile.MakeDirs(FLAGS.output_dir)
  if not tf.gfile.Exists(FLAGS.model_dir):
    tf.gfile.MakeDirs(FLAGS.model_dir)
  if not tf.gfile.Exists(FLAGS.init_checkpoint): FLAGS.init_checkpoint = None

  processors = {
      "de-en": De2EnTranslateProcessor,
      "nlpcc2018+hsk": CorrectProcessor,
  }

  processor = processors[TaskType]()

  source_ntoken = len(processor.src_idx2word)
  target_ntoken = len(processor.trg_idx2word)
  with tf.gfile.Open(FLAGS.model_config_path, "w") as fp:
      json.dump({"source_ntoken": source_ntoken, "target_ntoken": target_ntoken}, fp, indent=4)

  sp = spm.SentencePieceProcessor()
  sp.Load(FLAGS.spiece_model_file)
  def tokenize_fn(text):
    text = preprocess_text(text, lower=FLAGS.uncased)
    return encode_ids(sp, text)

  run_config = model_utils.configure_tpu(FLAGS)

  model_fn = get_model_fn()

  spm_basename = os.path.basename(FLAGS.spiece_model_file)

  estimator = tf.estimator.Estimator(params={"source_ntoken": source_ntoken, "target_ntoken": target_ntoken},
        model_fn=model_fn,
        config=run_config)

  if FLAGS.do_train:
    train_file_base = "{}.len-{}.train.tf_record".format(TaskType, FLAGS.max_seq_length)
    train_file = os.path.join(FLAGS.output_dir, train_file_base)
    tf.logging.info("Use tfrecord file {}".format(train_file))

    train_examples = processor.get_train_examples()
    np.random.shuffle(train_examples)
    tf.logging.info("Num of train samples: {}".format(len(train_examples)))

    file_based_convert_examples_to_features(train_examples, FLAGS.max_seq_length, train_file, FLAGS.num_passes)

    train_input_fn = file_based_input_fn_builder(
        input_file=train_file,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True)

    estimator.train(input_fn=train_input_fn, max_steps=FLAGS.train_steps)

  if FLAGS.do_eval or FLAGS.do_predict:
    eval_examples = processor.get_dev_examples()

    tf.logging.info("Num of eval samples: {}".format(len(eval_examples)))

  if FLAGS.do_eval:
    eval_file_base = "{}.len-{}.{}.eval.tf_record".format(spm_basename, FLAGS.max_seq_length, FLAGS.eval_split)
    eval_file = os.path.join(FLAGS.output_dir, eval_file_base)

    file_based_convert_examples_to_features(eval_examples, FLAGS.max_seq_length, eval_file, FLAGS.num_passes)

    #assert len(eval_examples) % FLAGS.eval_batch_size == 0
    eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)

    eval_input_fn = file_based_input_fn_builder(input_file=eval_file, seq_length=FLAGS.max_seq_length, is_training=False, drop_remainder=True)

    # Filter out all checkpoints in the directory
    steps_and_files = []
    filenames = tf.gfile.ListDirectory(FLAGS.model_dir)

    for filename in filenames:
      if filename.endswith(".index"):
        ckpt_name = filename[:-6]
        cur_filename = join(FLAGS.model_dir, ckpt_name)
        global_step = int(cur_filename.split("-")[-1])
        tf.logging.info("Add {} to eval list.".format(cur_filename))
        steps_and_files.append([global_step, cur_filename])
    steps_and_files = sorted(steps_and_files, key=lambda x: x[0])

    # Decide whether to evaluate all ckpts
    if not FLAGS.eval_all_ckpt:
      steps_and_files = steps_and_files[-1:]

    eval_results = []
    for global_step, filename in sorted(steps_and_files, key=lambda x: x[0]):
      ret = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps, checkpoint_path=filename)

      ret["step"] = global_step
      ret["path"] = filename

      eval_results.append(ret)

      tf.logging.info("=" * 80)
      log_str = "Eval result | "
      for key, val in sorted(ret.items(), key=lambda x: x[0]):
        log_str += "{} {} | ".format(key, val)
      tf.logging.info(log_str)

    eval_results.sort(key=lambda x: x["loss"], reverse=True)

    tf.logging.info("=" * 80)
    log_str = "Best result | "
    for key, val in sorted(eval_results[0].items(), key=lambda x: x[0]):
      log_str += "{} {} | ".format(key, val)
    tf.logging.info(log_str)

  if FLAGS.do_predict:
    eval_file_base = "{}.len-{}.{}.predict.tf_record".format(spm_basename, FLAGS.max_seq_length, FLAGS.eval_split)
    eval_file = os.path.join(FLAGS.output_dir, eval_file_base)

    file_based_convert_examples_to_features(eval_examples, FLAGS.max_seq_length, eval_file, FLAGS.num_passes)

    pred_input_fn = file_based_input_fn_builder(input_file=eval_file, seq_length=FLAGS.max_seq_length, is_training=False, drop_remainder=False)

    predict_results = []
    with tf.gfile.Open(os.path.join(predict_dir, "{}.tsv".format(TaskType)), "w") as fout:
      fout.write("cnt\tsource\ttarget\n")

      for pred_cnt, result in enumerate(estimator.predict(
          input_fn=pred_input_fn,
          yield_single_examples=True,
          checkpoint_path=FLAGS.predict_ckpt)):
        if pred_cnt % 1000 == 0:
          tf.logging.info("Predicting submission for example: {}".format(pred_cnt))

        pred = " ".join([str(e) for e in result["pred"]])
        output_ids = " ".join([str(e) for e in result["output_ids"]])
        predict_results.append({"pred": pred, "output_ids": output_ids})

        fout.write("{}\t{}\t{}\n".format(pred_cnt, pred, output_ids))

    predict_json_path = os.path.join(predict_dir, "{}.logits.json".format(TaskType))

    with tf.gfile.Open(predict_json_path, "w") as fp:
      json.dump(predict_results, fp, indent=4)

# ********************************************** run model by session *************************************************#
FLAGS = tf.app.flags.FLAGS
class seq2seq:
    def __init__(self,FLAGS=FLAGS, is_training=False):
        tf.logging.set_verbosity(tf.logging.INFO)
        self.input_ids = tf.placeholder(dtype=tf.int64, shape=[None, FLAGS.max_seq_length], name="input_ids")
        self.input_mask = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.max_seq_length], name="input_mask")
        self.output_ids = tf.placeholder(dtype=tf.int64, shape=[None, FLAGS.max_seq_length], name="output_ids")
        self.output_mask = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.max_seq_length], name="output_mask")
        inp_ids = tf.transpose(self.input_ids, [1, 0])
        inp_mask = tf.transpose(self.input_mask, [1, 0])
        output_ids = self.output_ids
        output_mask = tf.transpose(self.output_mask, [1, 0])
        # define decoder inputs
        decoder_inputs = tf.concat((tf.ones_like(output_ids[:, :1]) * 2, output_ids[:, :-1]), -1)  # 2代表<S>，是decoder的初始输入
        decoder_inputs = tf.transpose(decoder_inputs, [1, 0])
        self.sess = tf.Session()
        with tf.gfile.Open(FLAGS.model_config_path) as fp:
            config_dict = json.load(fp)
        processor = Seq2SeqProcessor()
        self.en_word2ids, self.en_id2words = processor.en_word2idx, processor.en_idx2word
        self.de_word2ids = processor.de_word2idx
        args = dict(
            FLAGS=FLAGS,
            is_training=is_training,
            inp_ids=inp_ids,
            inp_mask=inp_mask,
            source_ntoken=config_dict.get("source_ntoken"),
            target_ntoken=config_dict.get("target_ntoken"),
            output_mask=output_mask,
            output_ids=output_ids,
            decoder_inputs=decoder_inputs)
        self.s2sm = seq2seq_models.seq2seqmodel(**args)
        num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
        tf.logging.info('#params: {}'.format(num_params))
        self.s2sm.saver.restore(self.sess, FLAGS.init_checkpoint)

    def translate(self, sentence):
        input_ids = [self.de_word2ids.get(word, 1) for word in (sentence + u" </S>").split()]  # 1: OOV, </S>: End of Text
        if len(input_ids) > FLAGS.max_seq_length: input_ids = input_ids[:FLAGS.max_seq_length]
        input_mask = [0] * len(input_ids)
        if len(input_ids) < FLAGS.max_seq_length:
            delta_len = FLAGS.max_seq_length - len(input_ids)
            input_ids = [0] * delta_len + input_ids
            input_mask = [1] * delta_len + input_mask
        ### Autoregressive inference
        ### 在测试的时候是一个一个预测
        preds = np.zeros((1, FLAGS.max_seq_length), np.int32)
        output_mask = [[1] * FLAGS.max_seq_length]
        for i in range(FLAGS.max_seq_length):
            output_mask[0][i] = 0
            _preds, adebug = self.sess.run([self.s2sm.preds, self.s2sm.debug], {self.input_ids: [input_ids], self.input_mask: [input_mask],
                                                     self.output_ids: preds, self.output_mask: output_mask})
            preds[:, i] = _preds[:, i]
        for pred in preds:
            d=" ".join(self.en_id2words[idx] for idx in pred)
            got = " ".join(self.en_id2words[idx] for idx in pred).split("</S>")[0].strip()
            a=1


if __name__ == "__main__":
  #s2s = seq2seq(); s2s.translate("Sehen Sie all die unterschiedlichen Teile?")
  tf.app.run()