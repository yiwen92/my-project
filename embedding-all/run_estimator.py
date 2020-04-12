from config import conf, SEQ_LEN
import tensorflow as tf
from embedding import create_embed_encoder, tf_sim, tf_loss
from data_utils import gen_train_samples

# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode):
    # Build the neural network
    # Because Dropout have different behavior at training and prediction time, we
    # need to create 2 distinct computation graphs that still share the same weights.
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    # get semantic vector of input
    emb_a = create_embed_encoder(features['a_in'], is_training=is_training)
    emb_b = create_embed_encoder(features['b_in'], is_training=is_training, is_normal=False)
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
    x_train, y_train = gen_train_samples(conf.train_samples)
    # Build the Estimator
    model = tf.estimator.Estimator(model_fn)
    # Define the input function for training
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'a_in': x_train, 'b_in': y_train}, y=y_train,
        batch_size=conf.batch_size, num_epochs=None, shuffle=True)
    # Train the Model
    model.train(input_fn, steps=conf.num_steps)
    # save model
    feature_spec = {'a_in': tf.placeholder(dtype=tf.int32, shape=[None, SEQ_LEN], name='entity'), \
                    'b_in': tf.placeholder(dtype=tf.int32, shape=[None, None, SEQ_LEN], name='entity_list')}
    serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)
    model.export_savedmodel(conf.models_path + 'estimator', serving_input_receiver_fn)



if __name__ == "__main__":
    run()
    pass