import tensorflow as tf
import numpy as np
from data_utils import gen_train_samples
from config import conf, SEQ_LEN, VOCAB_SIZE, EMBEDDING_DIM

def rnn():
    max_step_cnt = tf.placeholder(tf.int32, ())
    batch_sz = tf.placeholder(tf.int32, (None))
    idx = tf.constant(0)
    z = tf.Variable(0, tf.int32)
    cell_input = tf.zeros((batch_sz), dtype=tf.int64)
    """
    # z 是变量，值未定，因此 out_ids.shape == (None, batch_sz), 这样 while_loop 循环过程中， out_ids 参数位的 tensor 的 shape 始终是 (None, batch_sz),
    shape 没有发生变化。loop_varias 几个参数都类似设置了，所以 while_loop 的 shape_invariants 参数才可以不用设置。
    这里 out_ids 是最为concat 的起始tensor的，因此令z == 0， 这样out_ids 相当于一个空数组一样
    """
    out_ids = tf.zeros((z, batch_sz), dtype=tf.int64)
    out_probs = tf.zeros((z, batch_sz, out_size))
    def cond(idx, *_):
        return idx < max_step_cnt
    def body(idx, cell_input, cell_state, out_ids, out_probs):
        with tf.variable_scope("rnn"):
            tf.get_variable_scope().reuse_variables()
            cur_out, cur_state = cell(cell_input, cell_state)
        # cur_out.shape == (batch_size, rnn_hid_size)
        cur_prob = tf.nn.softmax(tf.matmul(cur_out, softmax_w) + softmax_b)
        # cur_prob.shape == (batch_size, out_size)
        # 按概率抽样选取
        out = tf.multinomial(tf.log(cur_prob), 1)
        cur_id = tf.reshape(out, (-1,))
        # 按最大概率选取
        # cur_id = tf.argmax(cur_prob, axis=-1)
        cur_out_ids = tf.concat([out_ids, tf.expand_dims(cur_id, 0)], axis=0)
        cur_probs = tf.concat([out_probs, tf.expand_dims(cur_prob, 0)], axis=0)
        return idx + 1, cur_id, cur_state, cur_out_ids, cur_probs
    # init_state 是个复合类型，如果用shape_invariants指定其shape信息，会比较复杂
    wl = tf.while_loop(cond, body, loop_vars=[idx, cell_input, init_state, out_ids, out_probs])

def test():
    x_valid, y_valid = gen_train_samples(conf.valid_samples)

    a = y_valid[:2, :, :]
    input_x = tf.reshape(a, [-1, SEQ_LEN])
    lengths = tf.cast(tf.reduce_sum(tf.sign(tf.abs(tf.reshape(a, [-1, SEQ_LEN]))), reduction_indices=1), tf.int32)
    embedding = tf.get_variable('embedding', [VOCAB_SIZE, EMBEDDING_DIM])
    embedding_inputs = tf.nn.embedding_lookup(embedding, input_x)

    def cond(i, _length, _input, _output):
        return tf.less(i, tf.shape(_input)[1])
    def body(i, _length, _input, _output):
        return [i+1, _length, _input, _output.write(i, _input[:, _length[i], :])]
    Out = tf.TensorArray(size=0, dtype=tf.float32, dynamic_size=True)
    Input = tf.placeholder(tf.float32, [None, None, None], name='input_tensor')
    res = tf.while_loop(cond, body, [0, lengths, embedding_inputs, Out])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        fetch = sess.run({'res': res[-1].stack(), 'lenghts': lengths})
        pass
    pass;tf.w

def test1():
    i = tf.constant(0)#tf.get_variable("ii", dtype=tf.int32, shape=[], initializer=tf.ones_initializer())
    n = tf.constant(10)
    def cond(a, n):
        return a < n
    def body(a, n):
        a = a + 1
        return a, n
    a = tf.while_loop(cond, body, [i, n])
    with tf.Session() as sess:
        #tf.global_variables_initializer().run()
        res = sess.run(a)
        print(res)

if __name__ == "__main__":
    test()