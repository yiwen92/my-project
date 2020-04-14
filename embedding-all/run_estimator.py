from config import conf, SEQ_LEN, FLAGS
import tensorflow as tf
from embedding import create_embed_encoder, tf_sim, tf_loss
from data_utils import gen_train_samples, gen_train_input_fn
import model_utils

# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode, params):
    # Build the neural network
    # Because Dropout have different behavior at training and prediction time, we
    # need to create 2 distinct computation graphs that still share the same weights.
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    # get semantic vector of input
    emb_a, _ = create_embed_encoder(features['entity_ids'], is_training=is_training)
    emb_b, _ = create_embed_encoder(features['entity_ids_list'], is_training=is_training, is_normal=False)
    # Define loss and optimizer
    sim_op, sim_emb = tf_sim(emb_a, emb_b)
    loss_op = tf_loss(sim_op, sim_emb)
    train_op = tf.train.AdamOptimizer(learning_rate=conf.learning_rate).minimize(loss_op, global_step=tf.train.get_global_step())
    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, predicting...
    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions={'sim_op': sim_op})  # pred_classes})
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss_op,
        train_op=train_op,
        )
    return estim_specs

def run():
    # 设置日志的打印级别：把日志设置为INFO级别
    tf.logging.set_verbosity(tf.logging.INFO)
    # 得到训练数据
    x_train, y_train = gen_train_samples(FLAGS.train_samples)
    # 运行参数配置
    run_config = model_utils.configure_tpu(FLAGS)
    # Build the Estimator
    model = tf.estimator.Estimator(model_fn, params={"seq_len": SEQ_LEN}, config=run_config)
    # Define the input function for training
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'entity_ids': x_train, 'entity_ids_list': y_train}, y=y_train,
        batch_size=conf.batch_size, num_epochs=None, shuffle=True)
    # Define the input function based on tf.record file
    #input_fn = gen_train_input_fn(conf.train_samples)
    # Train the Model
    model.train(input_fn, steps=FLAGS.train_steps)
    # save model
    feature_spec = {'entity_ids': tf.placeholder(dtype=tf.int32, shape=[None, SEQ_LEN], name='entity'), \
                    'entity_ids_list': tf.placeholder(dtype=tf.int32, shape=[None, None, SEQ_LEN], name='entity_list')}
    serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)
    model.export_savedmodel(conf.models_path + 'estimator', serving_input_receiver_fn)


if __name__ == "__main__":
    run()
    pass