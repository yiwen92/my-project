import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers
from cnn_model import TCNNConfig, TextCNN
from rnn_model import rnn_net
from config import SEQ_LEN, MAX_NUM_NEG

C2 = 0.1
droprate = 0.5
embed_dim = 10
similarity_type = 'cosine'
mu_pos = 0.8
use_max_sim_neg = True
mu_neg = -0.4
C_emb = 0.8

def _create_tf_embed_nn(x_in, is_training, layer_sizes, name):
    # type: (tf.Tensor, tf.Tensor, List[int], Text) -> tf.Tensor
    """Create nn with hidden layers and name"""
    reg = tf.contrib.layers.l2_regularizer(C2)
    x = x_in
    for i, layer_size in enumerate(layer_sizes):
        x = tf.layers.dense(inputs=x,
                            kernel_initializer=initializers.xavier_initializer(),
                            units=layer_size,
                            activation=tf.nn.relu,
                            kernel_regularizer=reg)#,
                            #name='hidden_layer_{}_{}'.format(name, i))
        x = tf.layers.dropout(x, rate=droprate, training=is_training)

    x = tf.layers.dense(inputs=x,
                        kernel_initializer=initializers.xavier_initializer(),
                        units=embed_dim,
                        kernel_regularizer=reg)#,
                        #name='embed_layer_{}'.format(name))
    return x

class Encoder:
    def __init__(self, model_type='rnn'):
        self.debug_rnn = []
        self.debug_sim = []
        pass

    def create_tf_embed(self, a_in,  # type: tf.Tensor
                         b_in,  # type: tf.Tensor
                         is_training  # type: tf.Tensor
                         ):
        # type: (...) -> Tuple[tf.Tensor, tf.Tensor]
        """Create tf graph for training"""
        self.emb_b = self.create_embed_encoder(b_in, is_training, False)
        self.emb_a = self.create_embed_encoder(a_in, is_training)
        #emb_a = _create_tf_embed_nn(a_in, is_training, [10], name='a')
        #emb_b = _create_tf_embed_nn(b_in, is_training, [10], name='b')
        return self.emb_a, self.emb_b

    def create_embed_encoder(self, x_in, is_training, is_normal=True):
        if is_normal:
            embedding = rnn_net(x_in, is_training)
        else:
            tmp = []
            for i in range(1 + MAX_NUM_NEG):
                single_embedding = rnn_net(x_in[:, i, :], is_training)   ;self.debug_rnn.append((x_in[:,i,:], single_embedding))
                tmp.append(single_embedding)
            embedding = tf.stack(tmp, 1)
        return embedding

    def tf_sim(self, a, b):
        # type: (tf.Tensor, tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]
        """Define similarity in two cases:
            sim: between embedded words and embedded intent labels
            sim_emb: between individual embedded intent labels only"""
        """
        a: [A]
        b: [B0+, B1-, B2-, B3-, ..., Bk-]
        """
        if similarity_type == 'cosine':
            # normalize embedding vectors for cosine similarity
            a = tf.nn.l2_normalize(a, -1)
            b = tf.nn.l2_normalize(b, -1)

        if similarity_type in {'cosine', 'inner'}:
            sim = tf.reduce_sum(tf.expand_dims(a, 1) * b, -1)          # sim([A], [B0+, B1-, B2-, B3-, ..., Bk-]) -> [sim(A, B0+), sim(A, B1-), sim(A, B2-), ..., sim(A, Bk-)]
            sim_emb = tf.reduce_sum(b[:, 0:1, :] * b[:, 1:, :], -1)    # sim([B0+], [B1-, B2-, B3-, ..., Bk-]) -> [sim(B0+, B1-), sim(B0+, B2-), ..., sim(B0+, Bk-)]
            self.debug_sim.append((a, tf.expand_dims(a, 1)))
            return sim, sim_emb

        else:
            raise ValueError("Wrong similarity type {}, should be 'cosine' or 'inner'".format(similarity_type))

    def tf_loss(self, sim, sim_emb):
        # type: (tf.Tensor, tf.Tensor) -> tf.Tensor
        """Define loss"""
        """
        a: [A]
        b: [B0+, B1-, B2-, B3-, ..., Bk-]
        """
        # loss for maximizing similarity with correct action
        loss = tf.maximum(0., mu_pos - sim[:, 0])           # loss += max(0, mu_pos - sim(A, B0+))

        if use_max_sim_neg:
            # minimize only maximum similarity over incorrect actions
            max_sim_neg = tf.reduce_max(sim[:, 1:], -1)
            loss += tf.maximum(0., mu_neg + max_sim_neg)    # loss += max(0, mu_neg + max([sim(A, B1-), sim(A, B2-), ..., sim(A, Bk-)]))
        else:
            # minimize all similarities with incorrect actions
            max_margin = tf.maximum(0., mu_neg + sim[:, 1:])
            loss += tf.reduce_sum(max_margin, -1)

        # penalize max similarity between intent embeddings
        max_sim_emb = tf.maximum(0., tf.reduce_max(sim_emb, -1))
        loss += max_sim_emb * C_emb                             # loss += C_emb * (max(0, max([sim(B0+, B1-), sim(B0+, B2-), ..., sim(B0+, Bk-)])))

        # average the loss over the batch and add regularization losses
        loss = (tf.reduce_mean(loss) + tf.losses.get_regularization_loss())
        return loss