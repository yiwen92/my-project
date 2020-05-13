import tensorflow as tf
from config import conf

if conf.model_type == 'nn':
    from nn_model import nn_net as encode_net
elif conf.model_type == 'cnn':
    from cnn_model import cnn_net as encode_net
elif conf.model_type == 'rnn':
    from rnn_model import rnn_net as encode_net
elif conf.model_type == 'atten':
    from attention_model import attention_net as encode_net
else:
    raise NotImplementedError("model type: {}".format(conf.model_type))

class Encoder:
    def __init__(self, model_type='rnn'):
        self.model_type = model_type
        self.debug_sim = []

    def create_tf_embed(self, a_in, b_in, is_training):
        """
        Create semantic encoder
        :param a_in: int32 Tensor in shape [batch_size, seq_len], the input token IDs.
        :param b_in: int32 Tensor in shape [batch_size, 1 + num_neg, seq_len], the input token IDs.
        :param is_training: bool, whether in training mode.
        :return: embedding of input, a_in -> [batch_size, semantic_dim], b_in -> [batch_size, 1 + num_neg, semantic_dim]
        """
        self.emb_a, self.debug_info_a = create_embed_encoder(a_in, is_training)
        self.emb_b, self.debug_info_b = create_embed_encoder(b_in, is_training, False)
        self.sim_ab = cal_sim(self.emb_a, self.emb_b)
        return self.emb_a, self.emb_b

def create_embed_encoder(x_in, is_training, is_normal=True):
    if is_normal:
        embedding, debug_info = encode_net(x_in, is_training)               # 以序列为单位进行编码：[batch_size, seq_len]
    else:
        x_size = tf.shape(x_in)
        x_in_dim = x_in.get_shape().as_list()[2]
        x_in_reshape = tf.reshape(x_in, [-1, x_in_dim])                         # x_in：[batch_size, (1 + num_neg), seq_len]
        x_embedding, debug_info = encode_net(x_in_reshape, is_training)         # 以序列为单位进行编码：[batch_size * (1 + num_neg), seq_len]
        x_embedding_dim = x_embedding.get_shape().as_list()[1]
        embedding = tf.reshape(x_embedding, [-1, x_size[1], x_embedding_dim])
        '''
        tmp = []; debug_info['emb_diff_'] = []
        for i in range(x_in.get_shape().as_list()[1]):
            single_embedding, _ = rnn_net(x_in[:, i, :], is_training)   ;debug_info['emb_diff_'].append((x_in[:,i,:], single_embedding))
            tmp.append(single_embedding)
        embedding1 = tf.stack(tmp, 1)   ; debug_info['embedding'] = embedding; debug_info['embedding1'] = embedding1
        '''
    return embedding, debug_info

def cal_sim(a, b):
    a = tf.nn.l2_normalize(a, -1)
    b = tf.nn.l2_normalize(b, -1)
    sim = tf.reduce_sum(tf.expand_dims(a, 1) * b, -1)
    return sim

def tf_sim(a, b, debug_sim=[]):
    """
    Define similarity in two cases:
        sim: between embedded words and embedded intent labels
        sim_emb: between individual embedded intent labels only
        a: [A]
        b: [B0+, B1-, B2-, B3-, ..., Bk-]
    :param a: [batch_size, semantic_dim]
    :param b: [batch_size, 1 + num_neg, semantic_dim]
    :param debug_sim:
    :return: sim: [batch_size, 1 + num_neg], sim_emb: [batch_size, num_neg]
    """
    if conf.similarity_type == 'cosine':
        # normalize embedding vectors for cosine similarity
        a = tf.nn.l2_normalize(a, -1)
        b = tf.nn.l2_normalize(b, -1)

    if conf.similarity_type in {'cosine', 'inner'}:
        sim = tf.reduce_sum(tf.expand_dims(a, 1) * b, -1)          # sim([A], [B0+, B1-, B2-, B3-, ..., Bk-]) -> [sim(A, B0+), sim(A, B1-), sim(A, B2-), ..., sim(A, Bk-)]
        sim_emb = tf.reduce_sum(b[:, 0:1, :] * b[:, 1:, :], -1)    # sim([B0+], [B1-, B2-, B3-, ..., Bk-]) -> [sim(B0+, B1-), sim(B0+, B2-), ..., sim(B0+, Bk-)]

        debug_sim.append((a, tf.expand_dims(a, 1)))

        return sim, sim_emb

    else:
        raise ValueError("Wrong similarity type {}, should be 'cosine' or 'inner'".format(conf.similarity_type))

def tf_loss(sim, sim_emb):
    """
    Define loss
        a: [A]
        b: [B0+, B1-, B2-, B3-, ..., Bk-]
    :param sim: [batch_size, 1 + num_neg]
    :param sim_emb: [batch_size, num_neg]
    :return: loss
    """
    # loss for maximizing similarity with correct action
    loss = tf.maximum(0., conf.mu_pos - sim[:, 0])           # loss += max(0, mu_pos - sim(A, B0+))

    if conf.use_max_sim_neg:
        # minimize only maximum similarity over incorrect actions
        max_sim_neg = tf.reduce_max(sim[:, 1:], -1)
        loss += tf.maximum(0., conf.mu_neg + max_sim_neg)    # loss += max(0, mu_neg + max([sim(A, B1-), sim(A, B2-), ..., sim(A, Bk-)]))
    else:
        # minimize all similarities with incorrect actions
        max_margin = tf.maximum(0., conf.mu_neg + sim[:, 1:])
        loss += tf.reduce_sum(max_margin, -1)

    # penalize max similarity between intent embeddings
    max_sim_emb = tf.maximum(0., tf.reduce_max(sim_emb, -1))
    loss += max_sim_emb * conf.C_emb                             # loss += C_emb * (max(0, max([sim(B0+, B1-), sim(B0+, B2-), ..., sim(B0+, Bk-)])))

    # average the loss over the batch and add regularization losses
    loss = (tf.reduce_mean(loss) + tf.losses.get_regularization_loss())
    return loss

def cross_entropy_loss(cos_sim, label, debug_info={}):
    entropy = tf.nn.softmax_cross_entropy_with_logits(logits=cos_sim, labels=label)
    loss = tf.reduce_mean(entropy)
    debug_info['cos_sim']=cos_sim;debug_info['label']=label;debug_info['entropy']=entropy;debug_info['loss']=loss
    return loss

def log_loss(cos_sim, debug_info={}):
    prob = tf.nn.softmax(cos_sim)
    hit_prob = tf.slice(prob, [0, 0], [-1, 1])
    loss = -tf.reduce_mean(tf.log(hit_prob))
    debug_info['cos_sim']=cos_sim;debug_info['prob']=prob;debug_info['hit_prob']=hit_prob;debug_info['loss']=loss
    return loss

def multi_loss(sim, sim_emb, label, debug_info={}):
    sim_origin = tf_loss(sim, sim_emb)
    sim_entrop = cross_entropy_loss(sim, label)
    sim_log = log_loss(sim)
    loss = 0.3 * sim_origin + 0.4 * sim_entrop + 0.5 * sim_log
    debug_info['sim']=sim;debug_info['sim_emb']=sim_emb;debug_info['label']=label;debug_info['sim_origin']=sim_origin
    debug_info['sim_entrop']=sim_entrop; debug_info['sim_log']=sim_log;debug_info['loss']=loss
    return loss