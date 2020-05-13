import xlnet
from config import FLAGS
import tensorflow as tf

class AttentionConfig(object):
    """Attention 配置参数"""
    pass

def attention_net(input_x, is_training=True, scope='AttenNet', config=AttentionConfig()):
    debug_info = {}
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        bsz = tf.shape(input_x)[0]
        qlen = tf.shape(input_x)[1]
        segment_ids = tf.zeros([bsz, qlen], dtype=tf.int32)
        used = tf.sign(tf.abs(input_x))
        length = tf.reduce_sum(used, reduction_indices=1)
        lengths = tf.cast(length, tf.int32)
        # attention mask
        def cond(i, _length, _output):
            return tf.less(i, tf.shape(_length)[0])
        def body(i, _length, _output):
            return [i + 1, _length, _output.write(i, tf.concat([tf.zeros([_length[i]]), tf.ones(qlen - _length[i])], axis=-1))]
        Out = tf.TensorArray(size=0, dtype=tf.float32, dynamic_size=True, clear_after_read=False)
        res = tf.while_loop(cond, body, [0, lengths, Out])
        input_mask = tf.convert_to_tensor(res[-1].stack())
        inp = tf.transpose(input_x, [1, 0])
        seg_id = tf.transpose(segment_ids, [1, 0])
        inp_mask = tf.transpose(input_mask, [1, 0])
        # XLNetConfig contains hyperparameters that are specific to a model checkpoint.
        xlnet_config = xlnet.XLNetConfig(json_path=FLAGS.model_config_path)
        # RunConfig contains hyperparameters that could be different between pretraining and finetuning.
        run_config = xlnet.create_run_config(is_training=is_training, is_finetune=True, FLAGS=FLAGS)
        # Construct an XLNet model
        xlnet_model = xlnet.XLNetModel(
            xlnet_config=xlnet_config,
            run_config=run_config,
            input_ids=inp,
            seg_ids=seg_id,
            input_mask=inp_mask)
        # Get a summary of the sequence using the last hidden state
        summary = xlnet_model.get_pooled_out(FLAGS.summary_type, FLAGS.use_summ_proj)
        # Get a sequence output
        seq_out = xlnet_model.get_sequence_output()
        debug_info['input_x']=input_x;debug_info['segment_ids']=segment_ids;debug_info['input_mask']=input_mask
        debug_info['summary']=summary
    return summary, debug_info

if __name__ == "__main__":
    pass