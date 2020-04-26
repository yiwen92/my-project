#coding:utf8
#curl -d '{"instances": [1.0, 2.0, 5.0]}' -X POST http://localhost:8501/v1/models/half_plus_two:predict

import json, requests, sys, pickle, heapq, random
import tensorflow.contrib.keras as kr
from data_utils import gen_train_samples
from config import FLAGS
import numpy as np
import tensorflow as tf

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# 原始的样本数据
valid_data = gen_train_samples(FLAGS.valid_samples)
entity_ids, entity_ids_list = np.array(valid_data[0], dtype="int32"), np.array(valid_data[1], dtype="int32")
# 人造的数据
entity_ids = np.random.randint(0, 1000, size=[6, 10])
entity_ids_list = np.random.randint(0, 1000, size=[6, 15, 10])

sen_x_pad = [random.sample([i for i in range(1000)], 200)]
#sen_data = json.dumps({"signature_name": "serving_default", "instances": sen_x_pad})
sen_data = json.dumps({"signature_name": "serving_default", "instances": [{"input": sen_x_pad}]})
sen_data = json.dumps({"instances": sen_x_pad})
sen_json_response = requests.post('http://192.168.7.218:8511/v1/models/sentence_class:predict', data=sen_data)

ent_data = json.dumps({"instances": [{"entity_ids": entity_ids[0], "entity_ids_list": entity_ids_list[0]}, \
    {"entity_ids": entity_ids[1], "entity_ids_list": entity_ids_list[1]}]}, cls=NumpyEncoder)
json_response = requests.post('http://192.168.7.218:8512/v1/models/embedding_entity:predict', data=ent_data)
predictions = json.loads(json_response.text)['predictions']

a=1

if __name__ == '__main__':
    pass