#coding:utf8
from __future__ import division, print_function, absolute_import
from data.loader import build_vocab, read_category, read_vocab, process_file
import tensorflow as tf
import pickle, os

train_dir = './data/sen_class.train2'
test_dir = './data/sen_class.test2'
val_dir = './data/sen_class.val2'
vocab_dir = './data/sen_class.vocab2'
model_path = './model/'
map_path = model_path + 'id.maps'

if not os.path.exists(model_path): os.makedirs(model_path)

# Training Parameters
learning_rate = 0.001
num_steps = 20
batch_size = 128
seq_length = 100    # data input length

# Network Parameters
embedding_dim = 20 # 词向量维度
dropout = 0.5 # Dropout, probability to drop a unit
num_filters = 25  # 卷积核数目
kernel_size = 5  # 卷积核尺寸
hidden_dim = 128  # 全连接层神经元

# data process
labels = build_vocab(train_dir, test_dir, val_dir, vocab_dir)
num_classes = len(labels)
categories, cat_to_id = read_category(labels)
words, word_to_id = read_vocab(vocab_dir)
vocab_size = len(words)
x_train, y_train_ = process_file(train_dir, word_to_id, cat_to_id, seq_length);   y_train = y_train_.argmax(axis=1)
x_test, y_test_ = process_file(test_dir, word_to_id, cat_to_id, seq_length);   y_test = y_test_.argmax(axis=1)
with open(map_path, "wb") as f:  pickle.dump([word_to_id, cat_to_id, seq_length, num_classes], f)

# Create the neural network
def conv_net(x_dict, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        embedding = tf.get_variable('embedding', [vocab_size, embedding_dim])
        # TF Estimator input is a dict, in case of multiple inputs
        input_x = x_dict['input']
        embedding_inputs = tf.nn.embedding_lookup(embedding, input_x)

        conv = tf.layers.conv1d(embedding_inputs, num_filters, kernel_size, name='conv')
        gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')

        fc = tf.layers.dense(gmp, hidden_dim, name='fc1')
        fc = tf.layers.dropout(fc, rate=dropout, training=is_training)
        fc = tf.nn.relu(fc)

        logits = tf.layers.dense(fc, n_classes, name='fc2')

    return logits


# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode):
    # Build the neural network
    # Because Dropout have different behavior at training and prediction time, we
    # need to create 2 distinct computation graphs that still share the same weights.
    logits_train = conv_net(features, num_classes, dropout, reuse=False,
                            is_training=True)
    logits_test = conv_net(features, num_classes, dropout, reuse=True,
                           is_training=False)

    # Predictions
    pred_classes = tf.argmax(tf.nn.softmax(logits_test), 1)  # 预测类别
    pred_probas = tf.nn.softmax(logits_test)
    prob_class = tf.nn.top_k(pred_probas, 3)

    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions={'output': prob_class})#pred_classes})

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)))
    # 优化器
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})

    return estim_specs

def run():
    # Build the Estimator
    model = tf.estimator.Estimator(model_fn)

    # Define the input function for training
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'input': x_train}, y=y_train,
        batch_size=batch_size, num_epochs=None, shuffle=True)
    # Train the Model
    model.train(input_fn, steps=num_steps)

    # Evaluate the Model
    # Define the input function for evaluating
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'input': x_test}, y=y_test,
        batch_size=batch_size, shuffle=False)
    # Use the Estimator 'evaluate' method
    e = model.evaluate(input_fn)

    print("Testing Accuracy:", e['accuracy'])

    # save model
    feature_spec = {'input': tf.placeholder(dtype=tf.int32, shape=[None, seq_length], name='feat_input')}
    serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)
    model.export_savedmodel(model_path, serving_input_receiver_fn)

if __name__ == '__main__':
    run()
    aa=1