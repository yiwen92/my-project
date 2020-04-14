import json,re,math,random,time,os,shutil,sys
import numpy as np
import tensorflow as tf
import gensim.models.word2vec as w2v
from tensorflow.contrib.rnn import GRUCell
from tensorflow.nn import static_bidirectional_rnn as bi_rnn

W2V = w2v.Word2Vec.load("./res/w2v.mdl")
UNK = W2V.wv.get_vector("。").tolist()

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('max_steps', 1200000, 'Number of steps to run trainer.')
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
tf.app.flags.DEFINE_integer("bs", 2000, "Batch size.")
tf.app.flags.DEFINE_integer("negnum", 100, "Number of negative samples.")
tf.app.flags.DEFINE_string("summaries_dir", "log/titleVsdesc.lstm/", "Summary dir.")
tf.app.flags.DEFINE_integer("vs", len(UNK), "Vector size.")

tf.app.flags.DEFINE_integer("ebdnum", 128, "Demension of embedding.")

TI_SEQ_LEN=3
DE_SEQ_LEN=34
HIDDEN_SIZE=64
T_ATTENTION_SIZE= 2
C_ATTENTION_SIZE= 10

def grabSamples(f):
    np.random.seed(int(time.time()))
    title,cont = [],[]
    while True:
        l = f.readline()
        if not l:
            f.seek(0)
            continue

        if np.random.randint(0,1000)%3 !=0:continue

        try:
            [ti, desc]= json.loads(l)
            ti, desc = ti[:TI_SEQ_LEN], desc[:DE_SEQ_LEN]
            for _ in range(len(ti), TI_SEQ_LEN):ti.append(UNK)
            for _ in range(len(desc), DE_SEQ_LEN):desc.append(UNK)

            if len(title) < FLAGS.bs:title.append(np.array(ti))
            cont.append(np.array(desc))

            if len(cont) == FLAGS.bs*(FLAGS.negnum+1):return title,cont, f.tell()

        except Exception as e:
            print("[EXCEPTION]grabSamples",e)

def attention(inputs, attention_size, time_major=False, return_alphas=False):
    if isinstance(inputs, tuple):
        print("========= tuple")
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,D) => (B,T,D) B: batch size, T:Sequence length, D:hidden size of the RNN layer
        inputs = tf.transpose(inputs, [1, 0, 2])

    hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer

    # Trainable parameters
    w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    with tf.name_scope('v'):
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
    alphas = tf.nn.softmax(vu, name='alphas')         # (B,T) shape

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    if not return_alphas:
        return output
    else:
        return output, alphas

FW = GRUCell(HIDDEN_SIZE)
BW = GRUCell(HIDDEN_SIZE)

with tf.name_scope('Title_rnn'):
    t_btc = tf.placeholder(tf.float32,
                           shape=[None,TI_SEQ_LEN, FLAGS.vs],
                           name='TitleBatch')

    rnn,_,_ = bi_rnn(FW, BW,
                     inputs=tf.unstack(t_btc, TI_SEQ_LEN, axis=1),
                     dtype=tf.float32)

    smyt0 = tf.summary.histogram('RNN_outputs', rnn)

with tf.name_scope('Title_Attention'):
    ty, alphas = attention(rnn, T_ATTENTION_SIZE, True, return_alphas=True)
    ty = tf.identity(ty, name="emb")
    smyt1 = tf.summary.histogram('alphas', alphas)

with tf.name_scope('Content_rnn'):
    c_btc = tf.placeholder(tf.float32,
                           shape=[None,DE_SEQ_LEN, FLAGS.vs],
                           name='ContentBatch')

    rnn1,_,_ = bi_rnn(FW, BW,
                      inputs=tf.unstack(c_btc, DE_SEQ_LEN, axis=1),
                      dtype=tf.float32)

    smyc0 = tf.summary.histogram('RNN_outputs', rnn1)

with tf.name_scope('Content_Attention'):
    cy, alphas = attention(rnn1, C_ATTENTION_SIZE, True, return_alphas=True)
    cy = tf.identity(cy, name="emb")
    smyc1 = tf.summary.histogram('alphas', alphas)

with tf.name_scope('Cosine_Similarity'):
    query_norm = tf.tile(tf.sqrt(tf.reduce_sum(tf.square(ty), 1, True)), [FLAGS.negnum + 1, 1])
    doc_norm = tf.sqrt(tf.reduce_sum(tf.square(cy), 1, True))

    prod = tf.reduce_sum(tf.multiply(tf.tile(ty, [FLAGS.negnum + 1, 1]), cy), 1, True)
    norm_prod = tf.multiply(query_norm, doc_norm)

    cos_sim_raw = tf.truediv(prod, norm_prod)

    cos_sim = tf.transpose(tf.reshape(tf.transpose(cos_sim_raw), [FLAGS.negnum + 1, FLAGS.bs])) * 20

with tf.name_scope('Loss'):
    prob = tf.nn.softmax((cos_sim))
    hit_prob = tf.slice(prob, [0, 0], [-1, 1])
    loss = -tf.reduce_sum(tf.log(hit_prob)) / FLAGS.bs
    loss_summary = tf.summary.scalar('loss', loss)

with tf.name_scope('Training'):
    train_step = tf.train.AdamOptimizer(FLAGS.learning_rate,name='Adam').minimize(loss)

saver = tf.train.Saver()

merged = tf.summary.merge_all()

config = tf.ConfigProto(device_count={"CPU": 8})
config.gpu_options.allow_growth = True


with tf.Session(config=config) as sess:
    saver.restore(sess, FLAGS.summaries_dir+"sess.ckpt")
    #sess.run(tf.global_variables_initializer())

    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir+ 'graph', sess.graph)

    f = open("log/til_desc_4lstm.txt", "r")
    f.seek(8234560000)
    pst = 0
    for step in range(19022, FLAGS.max_steps):
        start = time.time()
        f.seek(pst)
        ti,de,pst = grabSamples(f)
        if not ti:break

        assert(len(de) == FLAGS.bs*(FLAGS.negnum+1))
        assert(len(ti) == FLAGS.bs)

        dend = time.time()
        print("Batch data is ready...%ds"%int(dend-start))

        _, ls,sm0,sm1,sm2,sm3 = sess.run([train_step,loss_summary,smyt0,smyt1,smyc0,smyc1],
                         feed_dict={
                             t_btc : ti
                             ,c_btc   : de
                         }
                         )

        train_writer.add_summary(ls, step)
        train_writer.add_summary(sm0, step)
        train_writer.add_summary(sm1, step)
        train_writer.add_summary(sm2, step)
        train_writer.add_summary(sm3, step)

        end = time.time()

        save_path = saver.save(sess, FLAGS.summaries_dir+"sess.ckpt")

        if os.path.isdir(FLAGS.summaries_dir+"model"):shutil.rmtree(FLAGS.summaries_dir+'model')
        tf.saved_model.simple_save(sess,
                                FLAGS.summaries_dir+'model',
                                {"T":t_btc},
                                {"Y":ty}
                                )
        print("Epoch:%d|time: %ds"%(step, end-start))



    save_path = saver.save(sess, FLAGS.summaries_dir+"sess.ckpt")
    print("Model saved in file: ", save_path)

