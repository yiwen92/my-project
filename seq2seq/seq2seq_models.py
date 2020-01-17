import tensorflow as tf
import numpy as np

def head_projection(h, d_model, n_head, d_head, kernel_initializer, name):
  """Project hidden states to a specific head with a 4D-shape."""
  proj_weight = tf.get_variable('{}/kernel'.format(name), [d_model, n_head, d_head], dtype=h.dtype, initializer=kernel_initializer)
  head = tf.einsum('ibh,hnd->ibnd', h, proj_weight)
  return head

def rel_shift(x, klen=-1):
  """perform relative shift to form the relative attention score."""
  x_size = tf.shape(x)

  x = tf.reshape(x, [x_size[1], x_size[0], x_size[2], x_size[3]])
  x = tf.slice(x, [1, 0, 0, 0], [-1, -1, -1, -1])
  x = tf.reshape(x, [x_size[0], x_size[1] - 1, x_size[2], x_size[3]])
  x = tf.slice(x, [0, 0, 0, 0], [-1, klen, -1, -1])

  return x

def rel_attn_core(q_head, k_head_h, v_head_h, k_head_r, seg_embed, seg_mat, r_w_bias, r_r_bias, r_s_bias, attn_mask, dropatt, is_training, scale):
  """Core relative positional attention operations."""

  # content based attention score
  ac = tf.einsum('ibnd,jbnd->ijbn', q_head + r_w_bias, k_head_h)

  # position based attention score
  bd = tf.einsum('ibnd,jbnd->ijbn', q_head + r_r_bias, k_head_r)
  bd = rel_shift(bd, klen=tf.shape(ac)[1])

  # segment based attention score
  if seg_mat is None:
    ef = 0
  else:
    ef = tf.einsum('ibnd,snd->ibns', q_head + r_s_bias, seg_embed)
    ef = tf.einsum('ijbs,ibns->ijbn', seg_mat, ef)

  # merge attention scores and perform masking
  attn_score = (ac + bd + ef) * scale
  if attn_mask is not None:
    # attn_score = attn_score * (1 - attn_mask) - 1e30 * attn_mask
    attn_score = attn_score - 1e30 * attn_mask

  # attention probability
  attn_prob = tf.nn.softmax(attn_score, 1)
  attn_prob = tf.layers.dropout(attn_prob, dropatt, training=is_training)

  # attention output
  attn_vec = tf.einsum('ijbn,jbnd->ibnd', attn_prob, v_head_h)

  return attn_vec

def post_attention(h, attn_vec, d_model, n_head, d_head, dropout, is_training, kernel_initializer, residual=True):
  """Post-attention processing."""
  # post-attention projection (back to `d_model`)
  proj_o = tf.get_variable('o/kernel', [d_model, n_head, d_head], dtype=h.dtype, initializer=kernel_initializer)
  attn_out = tf.einsum('ibnd,hnd->ibh', attn_vec, proj_o)

  attn_out = tf.layers.dropout(attn_out, dropout, training=is_training)
  if residual:
    output = tf.contrib.layers.layer_norm(attn_out + h, begin_norm_axis=-1, scope='LayerNorm')
  else:
    output = tf.contrib.layers.layer_norm(attn_out, begin_norm_axis=-1, scope='LayerNorm')

  return output

def embedding_lookup(x, n_token, d_embed, initializer, scope='embedding', reuse=None, dtype=tf.float32):
  """TPU and GPU embedding_lookup function."""
  with tf.variable_scope(scope, reuse=reuse):
    lookup_table = tf.get_variable('lookup_table', [n_token, d_embed], dtype=dtype, initializer=initializer)
    return tf.nn.embedding_lookup(lookup_table, x), lookup_table

def rel_multihead_attn(q, kv, r, r_w_bias, r_r_bias, seg_mat, r_s_bias, seg_embed, attn_mask, d_model, n_head,
                       d_head, dropout, dropatt, is_training, kernel_initializer, scope='rel_attn', reuse=False):
  """Multi-head attention with relative positional encoding."""

  scale = 1 / (d_head ** 0.5)
  with tf.variable_scope(scope, reuse=reuse):

    # content heads
    q_head_h = head_projection(q, d_model, n_head, d_head, kernel_initializer, 'q')
    k_head_h = head_projection(kv, d_model, n_head, d_head, kernel_initializer, 'k')
    v_head_h = head_projection(kv, d_model, n_head, d_head, kernel_initializer, 'v')

    # positional heads
    k_head_r = head_projection(r, d_model, n_head, d_head, kernel_initializer, 'r')

    # core attention ops
    attn_vec = rel_attn_core(q_head_h, k_head_h, v_head_h, k_head_r, seg_embed, seg_mat, r_w_bias, r_r_bias, r_s_bias, attn_mask, dropatt, is_training, scale)

    # post processing
    output = post_attention(q, attn_vec, d_model, n_head, d_head, dropout, is_training, kernel_initializer)

  return output

def positional_embedding(pos_seq, inv_freq, bsz=None):
  sinusoid_inp = tf.einsum('i,d->id', pos_seq, inv_freq)
  pos_emb = tf.concat([tf.sin(sinusoid_inp), tf.cos(sinusoid_inp)], -1)
  pos_emb = pos_emb[:, None, :]

  if bsz is not None:
    pos_emb = tf.tile(pos_emb, [1, bsz, 1])

  return pos_emb

def relative_positional_encoding(klen, d_model, clamp_len, bsz=None, dtype=None):
  """create relative positional encoding."""
  freq_seq = tf.range(0, d_model, 2.0)
  if dtype is not None and dtype != tf.float32:
    freq_seq = tf.cast(freq_seq, dtype=dtype)
  inv_freq = 1 / (10000 ** (freq_seq / d_model))
  # beg, end = klen - 1, -1
  beg, end = klen, -1

  fwd_pos_seq = tf.range(beg, end, -1.0)
  if dtype is not None and dtype != tf.float32:
    fwd_pos_seq = tf.cast(fwd_pos_seq, dtype=dtype)
  if clamp_len > 0:
    fwd_pos_seq = tf.clip_by_value(fwd_pos_seq, -clamp_len, clamp_len)
  pos_emb = positional_embedding(fwd_pos_seq, inv_freq, bsz)

  return pos_emb

def gelu(x):
  """Gaussian Error Linear Unit.
  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    x: float Tensor to perform activation.
  Returns:
    `x` with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
  return x * cdf

def positionwise_ffn(inp, d_model, d_inner, dropout, kernel_initializer, activation_type='relu', scope='ff', is_training=True, reuse=False):
  """Position-wise Feed-forward Network."""
  if activation_type == 'relu':
    activation = tf.nn.relu
  elif activation_type == 'gelu':
    activation = gelu
  else:
    raise ValueError('Unsupported activation type {}'.format(activation_type))

  output = inp
  with tf.variable_scope(scope, reuse=reuse):
    output = tf.layers.dense(output, d_inner, activation=activation, kernel_initializer=kernel_initializer, name='layer_1')
    output = tf.layers.dropout(output, dropout, training=is_training, name='drop_1')
    output = tf.layers.dense(output, d_model, kernel_initializer=kernel_initializer, name='layer_2')
    output = tf.layers.dropout(output, dropout, training=is_training, name='drop_2')
    output = tf.contrib.layers.layer_norm(output + inp, begin_norm_axis=-1, scope='LayerNorm')
  return output

def get_transformer_inputs(inp_ids, input_mask, n_token, d_model, clamp_len, dropout, n_layer,
                           n_head, d_head, initializer, scope, is_training, tf_float=tf.float32):
    """创建 transformer 模块需要的输入：output_encoding, pos_emb, r_w_bias, r_r_bias, non_tgt_mask """
    bsz = tf.shape(inp_ids)[1]
    qlen = tf.shape(inp_ids)[0]
    mlen = 0
    klen = mlen + qlen
    with tf.variable_scope(scope):
        r_w_bias = tf.get_variable('r_w_bias', [n_layer, n_head, d_head], dtype=tf_float, initializer=initializer)
        r_r_bias = tf.get_variable('r_r_bias', [n_layer, n_head, d_head], dtype=tf_float, initializer=initializer)
    # data mask: input mask
    data_mask = input_mask[None]
    attn_mask = data_mask[:, :, :, None]
    attn_mask = tf.cast(attn_mask > 0, dtype=tf_float)
    non_tgt_mask = -tf.eye(qlen, dtype=tf_float)
    non_tgt_mask = tf.concat([tf.zeros([qlen, mlen], dtype=tf_float), non_tgt_mask], axis=-1)
    non_tgt_mask = tf.cast((attn_mask + non_tgt_mask[:, :, None, None]) > 0, dtype=tf_float)
    ##### Word embedding
    word_emb_k, lookup_table = embedding_lookup(x=inp_ids, n_token=n_token, d_embed=d_model, initializer=initializer, scope='word_embedding')
    ##### Positional encoding
    pos_emb = relative_positional_encoding(klen, d_model, clamp_len, bsz=bsz, dtype=tf_float)
    output_encoding = tf.layers.dropout(word_emb_k, dropout, training=is_training)
    pos_emb = tf.layers.dropout(pos_emb, dropout, training=is_training)
    return output_encoding, pos_emb, r_w_bias, r_r_bias, non_tgt_mask

def _get_initializer(FLAGS):
  """Get variable intializer."""
  if FLAGS.init == "uniform":
    initializer = tf.initializers.random_uniform(minval=-FLAGS.init_range, maxval=FLAGS.init_range, seed=None)
  elif FLAGS.init == "normal":
    initializer = tf.initializers.random_normal(stddev=FLAGS.init_std, seed=None)
  else:
    raise ValueError("Initializer {} not supported".format(FLAGS.init))
  return initializer

class seq2seqmodel:
    def __init__(self, FLAGS, is_training, inp_ids, inp_mask, source_ntoken,
                 decoder_inputs, output_mask, target_ntoken, output_ids):
        kernel_initializer = _get_initializer(FLAGS)        ; self.debug = []
        common_args = dict(
            d_model=FLAGS.d_model,
            initializer=kernel_initializer,
            n_layer=FLAGS.n_layer,
            dropout=FLAGS.dropout,
            is_training=is_training,
            n_head=FLAGS.n_head,
            d_head=FLAGS.d_head,
            dropatt=FLAGS.dropatt,
            d_inner=FLAGS.d_inner,
        )
        encoder_args = dict(
            inp_ids=inp_ids,
            input_mask=inp_mask,
            n_token=source_ntoken,
        )
        encoder_args.update(common_args)
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            encoder_embedding, encoder_mask = self.encoder(**encoder_args)
        decoder_args = dict(
            encoder_embedding=encoder_embedding,
            inp_ids=decoder_inputs,
            input_mask=output_mask,
            n_token=target_ntoken,
            encoder_mask=encoder_mask,
        )
        decoder_args.update(common_args)
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            decoder_embedding = self.decoder(**decoder_args)
        output_embedding = tf.transpose(decoder_embedding, [1, 0, 2])
        self.out = output_embedding
        # Final linear projection
        self.logits = tf.layers.dense(output_embedding, target_ntoken, kernel_initializer=kernel_initializer, name="logits")
        self.preds = tf.to_int32(tf.argmax(self.logits, dimension=-1))
        istarget = tf.to_float(tf.not_equal(output_ids, 0))
        acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, output_ids)) * istarget / (tf.reduce_sum(istarget)))
        one_hot_target = tf.one_hot(output_ids, target_ntoken, dtype=output_embedding.dtype)
        self.per_example_loss = -tf.reduce_sum(tf.nn.log_softmax(self.logits) * one_hot_target, -1)
        self.total_loss = tf.reduce_mean(self.per_example_loss)
        # 保存模型
        self.saver = tf.train.Saver(tf.global_variables())

    def encoder(self, inp_ids, input_mask, n_token, d_model, initializer, n_layer, dropout, d_inner, is_training,
            n_head, d_head, dropatt, clamp_len=-1, scope='transformer_encoder', ff_activation='relu'):
        output_encoding, pos_emb, r_w_bias, r_r_bias, non_tgt_mask = get_transformer_inputs(inp_ids, input_mask, n_token,
                                d_model, clamp_len, dropout, n_layer,n_head, d_head, initializer, scope, is_training)
        self.debug.extend([input_mask, non_tgt_mask])
        for i in range(n_layer):
            with tf.variable_scope('layer_{}'.format(i)):
                output_encoding = rel_multihead_attn(       # Self Multi-Head Attention + Residual Add & Layer Norm
                    q=output_encoding, kv=output_encoding,
                    r=pos_emb, r_w_bias=r_w_bias[i], r_r_bias=r_r_bias[i], seg_mat=None, r_s_bias=None, seg_embed=None, attn_mask=non_tgt_mask,
                    d_model=d_model, n_head=n_head, d_head=d_head, dropout=dropout, dropatt=dropatt, is_training=is_training, kernel_initializer=initializer)
                output_encoding = positionwise_ffn(         # Feed Forword + Residual Add & Layer Norm
                    inp=output_encoding, d_model=d_model, d_inner=d_inner, dropout=dropout,
                    kernel_initializer=initializer, activation_type=ff_activation, is_training=is_training)
        return output_encoding, non_tgt_mask

    def decoder(self, inp_ids, input_mask, n_token, d_model, initializer, n_layer, dropout, d_inner, is_training, encoder_embedding,
                encoder_mask, n_head, d_head, dropatt, clamp_len=-1, scope='transformer_decoder', ff_activation='relu'):
        output_encoding, pos_emb, r_w_bias, r_r_bias, non_tgt_mask = get_transformer_inputs(inp_ids, input_mask, n_token,
                                    d_model, clamp_len, dropout, n_layer, n_head, d_head, initializer, scope, is_training)
        self.debug.extend([input_mask, encoder_mask, non_tgt_mask])
        for i in range(n_layer):
            with tf.variable_scope('layer_{}'.format(i)):
                output_encoding = rel_multihead_attn(  # Self Multi-Head Attention + Residual Add & Layer Norm
                    q=output_encoding, kv=output_encoding,
                    r=pos_emb, r_w_bias=r_w_bias[i], r_r_bias=r_r_bias[i], seg_mat=None, r_s_bias=None, seg_embed=None,
                    attn_mask=non_tgt_mask,
                    d_model=d_model, n_head=n_head, d_head=d_head, dropout=dropout, dropatt=dropatt,
                    is_training=is_training, kernel_initializer=initializer)
                output_encoding = rel_multihead_attn(  # Multi-Head Attention + Residual Add & Layer Norm
                    q=output_encoding, kv=encoder_embedding,
                    r=pos_emb, r_w_bias=r_w_bias[i], r_r_bias=r_r_bias[i], seg_mat=None, r_s_bias=None, seg_embed=None,
                    attn_mask=encoder_mask,
                    d_model=d_model, n_head=n_head, d_head=d_head, dropout=dropout, dropatt=dropatt,
                    is_training=is_training, kernel_initializer=initializer)
                output_encoding = positionwise_ffn(  # Feed Forword + Residual Add & Layer Norm
                    inp=output_encoding, d_model=d_model, d_inner=d_inner, dropout=dropout,
                    kernel_initializer=initializer, activation_type=ff_activation, is_training=is_training)
        return output_encoding
