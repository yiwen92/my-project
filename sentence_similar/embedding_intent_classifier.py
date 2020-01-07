from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import io, random, json
import copy
import logging
import os
from tqdm import tqdm
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers

from typing import List, Text, Any, Optional, Dict, Tuple

import numpy as np
INTENT_RANKING_LENGTH = 10
try:
    import cPickle as pickle
except ImportError:
    import pickle

logger = logging.getLogger(__name__)

class EmbeddingIntentClassifier():
    name = "intent_classifier_tensorflow_embedding"
    provides = ["intent", "intent_ranking"]
    requires = ["text_features"]
    defaults = {
        # nn architecture
        # sizes of hidden layers before the embedding layer for input words
        # the number of hidden layers is thus equal to the length of this list
        "hidden_layers_sizes_a": [256, 128],
        # sizes of hidden layers before the embedding layer for intent labels
        # the number of hidden layers is thus equal to the length of this list
        "hidden_layers_sizes_b": [128, 64],

        # training parameters
        # initial and final batch sizes - batch size will be
        # linearly increased for each epoch
        "batch_size": [64, 256],
        # number of epochs
        "epochs": 300,

        # embedding parameters
        # dimension size of embedding vectors
        "embed_dim": 20,
        # how similar the algorithm should try
        # to make embedding vectors for correct intent labels
        "mu_pos": 0.8,  # should be 0.0 < ... < 1.0 for 'cosine'
        # maximum negative similarity for incorrect intent labels
        "mu_neg": -0.4,  # should be -1.0 < ... < 1.0 for 'cosine'
        # the type of the similarity
        "similarity_type": 'cosine',  # string 'cosine' or 'inner'
        # the number of incorrect intents, the algorithm will minimize
        # their similarity to the input words during training
        "num_neg": 20,
        # flag: if true, only minimize the maximum similarity for
        # incorrect intent labels
        "use_max_sim_neg": True,
        # set random seed to any int to get reproducible results
        # try to change to another int if you are not getting good results
        "random_seed": None,

        # regularization parameters
        # the scale of L2 regularization
        "C2": 0.002,
        # the scale of how critical the algorithm should be of minimizing the
        # maximum similarity between embeddings of different intent labels
        "C_emb": 0.8,
        # dropout rate for rnn
        "droprate": 0.2,

        # flag: if true, the algorithm will split the intent labels into tokens
        #       and use bag-of-words representations for them
        "intent_tokenization_flag": False,
        # delimiter string to split the intent labels
        "intent_split_symbol": '_',

        # visualization of accuracy
        # how often to calculate training accuracy
        "evaluate_every_num_epochs": 10,  # small values may hurt performance
        # how many examples to use for calculation of training accuracy
        "evaluate_on_num_examples": 10, #1000  # large values may hurt performance
        "split_ratio": 0.8
    }
    def __init__(self):
        self.similarity_type = self.defaults.get('similarity_type')
        self.num_neg = self.defaults.get('num_neg')
        self.random_seed = self.defaults.get('random_seed')
        self.hidden_layer_sizes = {}
        self.hidden_layer_sizes['a'] = self.defaults.get('hidden_layers_sizes_a')
        self.hidden_layer_sizes['b'] = self.defaults.get('hidden_layers_sizes_b')
        self.C2 = self.defaults.get('C2')
        self.droprate = self.defaults.get('droprate')
        self.embed_dim = self.defaults.get('embed_dim')
        self.mu_pos = self.defaults.get('mu_pos')
        self.use_max_sim_neg = self.defaults.get('use_max_sim_neg')
        self.mu_neg = self.defaults.get('mu_neg')
        self.C_emb = self.defaults.get('C_emb')
        self.evaluate_on_num_examples = self.defaults.get('evaluate_on_num_examples')
        self.evaluate_every_num_epochs = self.defaults.get('evaluate_every_num_epochs')
        self.epochs = self.defaults.get('epochs')
        self.batch_size = self.defaults.get('batch_size')
        self.split_ratio = self.defaults.get('split_ratio')

    # noinspection PyPep8Naming
    def _create_all_Y(self, size):
        # type: (int) -> np.ndarray
        """Stack encoded_all_intents on top of each other to create candidates for training examples to calculate training accuracy"""
        return np.stack([self.encoded_all_intents] * size)

    # tf helpers:
    def _create_tf_embed_nn(self, x_in, is_training, layer_sizes, name):
        # type: (tf.Tensor, tf.Tensor, List[int], Text) -> tf.Tensor
        """Create nn with hidden layers and name"""
        reg = tf.contrib.layers.l2_regularizer(self.C2)
        x = x_in
        for i, layer_size in enumerate(layer_sizes):
            x = tf.layers.dense(inputs=x,
                                kernel_initializer=initializers.xavier_initializer(),
                                units=layer_size,
                                activation=tf.nn.relu,
                                kernel_regularizer=reg,
                                name='hidden_layer_{}_{}'.format(name, i))
            x = tf.layers.dropout(x, rate=self.droprate, training=is_training)

        x = tf.layers.dense(inputs=x,
                            kernel_initializer=initializers.xavier_initializer(),
                            units=self.embed_dim,
                            kernel_regularizer=reg,
                            name='embed_layer_{}'.format(name))
        return x

    def _create_tf_embed(self,
                         a_in,  # type: tf.Tensor
                         b_in,  # type: tf.Tensor
                         is_training  # type: tf.Tensor
                         ):
        # type: (...) -> Tuple[tf.Tensor, tf.Tensor]
        """Create tf graph for training"""
        emb_a = self._create_tf_embed_nn(a_in, is_training, self.hidden_layer_sizes['a'], name='a')
        emb_b = self._create_tf_embed_nn(b_in, is_training, self.hidden_layer_sizes['b'], name='b')
        return emb_a, emb_b

    def _tf_sim(self, a, b):
        # type: (tf.Tensor, tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]
        """Define similarity in two cases:
            sim: between embedded words and embedded intent labels
            sim_emb: between individual embedded intent labels only"""

        if self.similarity_type == 'cosine':
            # normalize embedding vectors for cosine similarity
            a = tf.nn.l2_normalize(a, -1)
            b = tf.nn.l2_normalize(b, -1)

        if self.similarity_type in {'cosine', 'inner'}:
            sim = tf.reduce_sum(tf.expand_dims(a, 1) * b, -1)
            sim_emb = tf.reduce_sum(b[:, 0:1, :] * b[:, 1:, :], -1)

            return sim, sim_emb

        else:
            raise ValueError("Wrong similarity type {}, "
                             "should be 'cosine' or 'inner'"
                             "".format(self.similarity_type))

    def _tf_loss(self, sim, sim_emb):
        # type: (tf.Tensor, tf.Tensor) -> tf.Tensor
        """Define loss"""
        # loss for maximizing similarity with correct action
        loss = tf.maximum(0., self.mu_pos - sim[:, 0])

        if self.use_max_sim_neg:
            # minimize only maximum similarity over incorrect actions
            max_sim_neg = tf.reduce_max(sim[:, 1:], -1)
            loss += tf.maximum(0., self.mu_neg + max_sim_neg)
        else:
            # minimize all similarities with incorrect actions
            max_margin = tf.maximum(0., self.mu_neg + sim[:, 1:])
            loss += tf.reduce_sum(max_margin, -1)

        # penalize max similarity between intent embeddings
        max_sim_emb = tf.maximum(0., tf.reduce_max(sim_emb, -1))
        loss += max_sim_emb * self.C_emb

        # average the loss over the batch and add regularization losses
        loss = (tf.reduce_mean(loss) + tf.losses.get_regularization_loss())
        return loss

    # training helpers:
    def _create_batch_b(self, batch_pos_b, intent_ids):
        # type: (np.ndarray, np.ndarray) -> np.ndarray
        """Create batch of intents, where the first is correct intent
            and the rest are wrong intents sampled randomly"""

        batch_pos_b = batch_pos_b[:, np.newaxis, :]

        # sample negatives
        batch_neg_b = np.zeros((batch_pos_b.shape[0], self.num_neg,
                                batch_pos_b.shape[-1]))

        for b in range(batch_pos_b.shape[0]):
            # create negative indexes out of possible ones
            # except for correct index of b
            aaaaa_ = intent_ids[b]
            negative_indexes = [i for i in range(
                                    self.encoded_all_intents.shape[0])
                                if int(intent_ids[b][i]) != 1]
            #negs = np.random.choice(negative_indexes, size=self.num_neg)
            negs = random.sample(negative_indexes, self.num_neg)

            batch_neg_b[b] = self.encoded_all_intents[negs]

        return np.concatenate([batch_pos_b, batch_neg_b], 1)

    def _linearly_increasing_batch_size(self, epoch):
        # type: (int) -> int
        """Linearly increase batch size with every epoch.
            The idea comes from https://arxiv.org/abs/1711.00489"""
        if not isinstance(self.batch_size, list):
            return int(self.batch_size)

        if self.epochs > 1:
            return int(self.batch_size[0] +
                       epoch * (self.batch_size[1] -
                                self.batch_size[0]) / (self.epochs - 1))
        else:
            return int(self.batch_size[0])

    # noinspection PyPep8Naming
    def _train_tf(self,
                  X,  # type: np.ndarray
                  Y,  # type: np.ndarray
                  intents_for_X,  # type: np.ndarray
                  loss,  # type: tf.Tensor
                  is_training,  # type: tf.Tensor
                  train_op,  # type: tf.Tensor
                  Y_all
                  ):
        # type: (...) -> None
        """Train tf graph"""
        self.session.run(tf.global_variables_initializer())
        train_split = int(len(X) * self.split_ratio)
        X_train = X[: train_split]; X_test = X[train_split: ]; Y_all_train = Y_all[: train_split]
        Y_train = Y[: train_split]; Y_test = Y[train_split: ]
        intents_for_X_train = intents_for_X[: train_split]; intents_for_X_test = intents_for_X[train_split: ]

        if self.evaluate_on_num_examples:
            logger.info("Accuracy is updated every {} epochs"
                        "".format(self.evaluate_every_num_epochs))

        pbar = tqdm(range(self.epochs), desc="Epochs")
        train_acc = 0
        last_loss = 0
        for ep in pbar:
            indices = np.random.permutation(len(X_train))     #; indices = list(range(len(X_train)))

            batch_size = self._linearly_increasing_batch_size(ep)           #; batch_size = 1
            batches_per_epoch = (len(X_train) // batch_size +
                                 int(len(X_train) % batch_size > 0))

            ep_loss = 0
            for i in range(batches_per_epoch):
                end_idx = (i + 1) * batch_size
                start_idx = i * batch_size
                batch_a = X_train[indices[start_idx:end_idx]]
                batch_pos_b = Y_train[indices[start_idx:end_idx]]
                batch_pos_b_all = Y_all_train[indices[start_idx:end_idx]]
                intents_for_b = intents_for_X_train[indices[start_idx:end_idx]]
                # add negatives
                #batch_b = self._create_batch_b(batch_pos_b, intents_for_b)
                batch_b = self._create_batch_b(batch_pos_b, batch_pos_b_all)

                sess_out = self.session.run(
                        {'loss': loss, 'train_op': train_op, 'a':self.a_in,'b':self.b_in,'c':self.word_embed,'d':self.intent_embed},
                        feed_dict={self.a_in: batch_a,
                                   self.b_in: batch_b,
                                   is_training: True}
                )
                ep_loss += sess_out.get('loss') / batches_per_epoch

            if self.evaluate_on_num_examples:
                if (ep == 0 or
                        (ep + 1) % self.evaluate_every_num_epochs == 0 or
                        (ep + 1) == self.epochs):
                    train_acc = self._output_training_stat(X_test, intents_for_X_test,
                                                           is_training)
                    last_loss = ep_loss

                pbar.set_postfix({
                    "loss": "{:.3f}".format(ep_loss),
                    "acc": "{:.3f}".format(train_acc)
                })
            else:
                pbar.set_postfix({
                    "loss": "{:.3f}".format(ep_loss)
                })

        if self.evaluate_on_num_examples:
            logger.info("Finished training embedding classifier, "
                        "loss={:.3f}, train accuracy={:.3f}"
                        "".format(last_loss, train_acc))

    # noinspection PyPep8Naming
    def _output_training_stat(self, X, intents_for_X, is_training):
        # type: (np.ndarray, np.ndarray, tf.Tensor) -> np.ndarray
        """Output training statistics"""
        n = self.evaluate_on_num_examples
        ids = np.random.permutation(len(X))[:n]     ;a = X[ids]; b = X[ids].shape
        all_Y = self._create_all_Y(X[ids].shape[0])

        train_sim = self.session.run(self.sim_op,
                                     feed_dict={self.a_in: X[ids],
                                                self.b_in: all_Y,
                                                is_training: False})
        a = np.argmax(train_sim, -1); aa = intents_for_X[ids]
        train_acc = np.mean(np.argmax(train_sim, -1) == intents_for_X[ids])
        return train_acc

    # noinspection PyPep8Naming
    def train(self, training_examples):
        """Train the embedding intent classifier on a data set."""
        logging.info('Embedding intent classifier parameter: %s', json.dumps(self.defaults, ensure_ascii=False))
        distinct_intents = set([example.get("intent") for example in training_examples])
        intent_dict = {intent: idx for idx, intent in enumerate(sorted(distinct_intents))}
        if len(intent_dict) < 2:
            logger.error("Can not train an intent classifier. "
                         "Need at least 2 different classes. "
                         "Skipping training of intent classifier.")
            return

        self.inv_intent_dict = {v: k for k, v in intent_dict.items()}
        self.encoded_all_intents = np.eye(len(intent_dict))

        X = np.stack([e.get("text_features") for e in training_examples])
        intents_for_X = np.array([intent_dict[e.get("intent")] for e in training_examples])
        Y = np.stack([self.encoded_all_intents[intent_idx] for intent_idx in intents_for_X])
        Y_all = copy.deepcopy(Y)
        max_multi_intent_num = 1
        for i in range(len(training_examples)):
            if 'multi_intents' not in training_examples[i]: continue
            if len(training_examples[i]['multi_intents']) > max_multi_intent_num: max_multi_intent_num = len(training_examples[i]['multi_intents'])
            for e in training_examples[i]['multi_intents']:
                Y_all[i][intent_dict[e]] = 1

        # check if number of negatives is less than number of intents
        logger.debug("Check if num_neg {} is smaller than "
                     "number of intents {}, "
                     "else set num_neg to the number of intents - 1"
                     "".format(self.num_neg,
                               self.encoded_all_intents.shape[0]))
        self.num_neg = min(self.num_neg, self.encoded_all_intents.shape[0] - max_multi_intent_num)  #self.num_neg = 4
        logging.info('k-negative sampling number is {}'.format(self.num_neg))
        self.graph = tf.Graph()
        with self.graph.as_default():
            # set random seed
            np.random.seed(self.random_seed)
            tf.set_random_seed(self.random_seed)

            self.a_in = tf.placeholder(tf.float32, (None, X.shape[-1]), name='a')
            self.b_in = tf.placeholder(tf.float32, (None, None, Y.shape[-1]), name='b')

            is_training = tf.placeholder_with_default(False, shape=())

            (self.word_embed, self.intent_embed) = self._create_tf_embed(self.a_in, self.b_in, is_training)

            self.sim_op, sim_emb = self._tf_sim(self.word_embed, self.intent_embed)
            loss = self._tf_loss(self.sim_op, sim_emb)

            train_op = tf.train.AdamOptimizer().minimize(loss)

            # train tensorflow graph
            tf_config = tf.ConfigProto()
            tf_config.gpu_options.allow_growth = False
            tf_config.gpu_options.per_process_gpu_memory_fraction = 0.5
            self.session = tf.Session(config=tf_config)

            self._train_tf(X, Y, intents_for_X,
                           loss, is_training, train_op, Y_all)

    # process helpers
    # noinspection PyPep8Naming
    def _calculate_message_sim(self,
                               X,  # type: np.ndarray
                               all_Y  # type: np.ndarray
                               ):
        # type: (...) -> Tuple[np.ndarray, List[float]]
        """Load tf graph and calculate message similarities"""

        message_sim = self.session.run(self.sim_op,
                                       feed_dict={self.a_in: X,
                                                  self.b_in: all_Y})
        message_sim = message_sim.flatten()  # sim is a matrix

        intent_ids = message_sim.argsort()[::-1]
        message_sim[::-1].sort()

        if self.similarity_type == 'cosine':
            # clip negative values to zero
            message_sim[message_sim < 0] = 0
        elif self.similarity_type == 'inner':
            # normalize result to [0, 1] with softmax
            message_sim = np.exp(message_sim)
            message_sim /= np.sum(message_sim)

        # transform sim to python list for JSON serializing
        return intent_ids, message_sim.tolist()

    # noinspection PyPep8Naming
    def process(self, message, **kwargs):
        # type: (Message, Any) -> None
        """Return the most likely intent and its similarity to the input."""

        intent = {"name": None, "confidence": 0.0}
        intent_ranking = []

        if self.session is None:
            logger.error("There is no trained tf.session: "
                         "component is either not trained or "
                         "didn't receive enough training data")

        else:
            # get features (bag of words) for a message
            X = message.text_features.reshape(1, -1)

            # stack encoded_all_intents on top of each other
            # to create candidates for test examples
            all_Y = self._create_all_Y(X.shape[0])

            # load tf graph and session
            intent_ids, message_sim = self._calculate_message_sim(X, all_Y)

            # if X contains all zeros do not predict some label
            if X.any() and intent_ids.size > 0:
                intent = {"name": self.inv_intent_dict[intent_ids[0]],
                          "confidence": message_sim[0]}

                ranking = list(zip(list(intent_ids), message_sim))
                ranking = ranking[:INTENT_RANKING_LENGTH]
                intent_ranking = [{"name": self.inv_intent_dict[intent_idx],
                                   "confidence": score}
                                  for intent_idx, score in ranking]

        message.intent = intent
        message.intent_ranking = intent_ranking

    def persist(self, model_dir):
        # type: (Text) -> Dict[Text, Any]
        """Persist this model into the passed directory.
        Return the metadata necessary to load the model again."""
        if self.session is None:
            return {"classifier_file": None}

        checkpoint = os.path.join(model_dir, self.name + ".ckpt")

        try:
            os.makedirs(os.path.dirname(checkpoint))
        except OSError as e:
            # be happy if someone already created the path
            import errno
            if e.errno != errno.EEXIST:
                raise
        with self.graph.as_default():
            self.graph.clear_collection('message_placeholder')
            self.graph.add_to_collection('message_placeholder',
                                         self.a_in)

            self.graph.clear_collection('intent_placeholder')
            self.graph.add_to_collection('intent_placeholder',
                                         self.b_in)

            self.graph.clear_collection('similarity_op')
            self.graph.add_to_collection('similarity_op',
                                         self.sim_op)

            self.graph.clear_collection('word_embed')
            self.graph.add_to_collection('word_embed',
                                         self.word_embed)
            self.graph.clear_collection('intent_embed')
            self.graph.add_to_collection('intent_embed',
                                         self.intent_embed)

            saver = tf.train.Saver()
            saver.save(self.session, checkpoint)

        with io.open(os.path.join(
                model_dir,
                self.name + "_inv_intent_dict.pkl"), 'wb') as f:
            pickle.dump(self.inv_intent_dict, f)
        with io.open(os.path.join(
                model_dir,
                self.name + "_encoded_all_intents.pkl"), 'wb') as f:
            pickle.dump(self.encoded_all_intents, f)

        return {"classifier_file": self.name + ".ckpt"}

    @classmethod
    def load(self, model_dir=None):
        if model_dir:
            file_name = self.name + '.ckpt'
            checkpoint = os.path.join(model_dir, file_name)
            graph = tf.Graph()
            with graph.as_default():
                self.session = tf.Session()
                saver = tf.train.import_meta_graph(checkpoint + '.meta')

                saver.restore(self.session, checkpoint)

                self.a_in = tf.get_collection('message_placeholder')[0]
                self.b_in = tf.get_collection('intent_placeholder')[0]

                self.sim_op = tf.get_collection('similarity_op')[0]

                word_embed = tf.get_collection('word_embed')[0]
                intent_embed = tf.get_collection('intent_embed')[0]

            with io.open(os.path.join(
                    model_dir,
                    self.name + "_inv_intent_dict.pkl"), 'rb') as f:
                self.inv_intent_dict = pickle.load(f)
            with io.open(os.path.join(
                    model_dir,
                    self.name + "_encoded_all_intents.pkl"), 'rb') as f:
                self.encoded_all_intents = pickle.load(f)
        else:
            logger.warning("Failed to load nlu model. Maybe path {} "
                           "doesn't exist"
                           "".format(os.path.abspath(model_dir)))