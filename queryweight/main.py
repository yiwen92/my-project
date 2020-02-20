import tensorflow as tf
from data_utils import create_data
from config import TASK
from absl import app
from train import main as pre_train_main
from handle_data import get_corpus

tf.logging.set_verbosity(tf.logging.INFO)

def get_pretrain_data():
    tf.app.run(create_data)

def pre_train_model():
    app.run(pre_train_main)

if __name__ == "__main__":
    if TASK == 0: get_corpus()                  # handle_data.py
    elif TASK == 1: get_pretrain_data()         # data_utils.py
    elif TASK == 2: pre_train_model()           # train.py
    else: raise NotImplementedError