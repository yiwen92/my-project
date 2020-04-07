#coding:utf8
#curl -d '{"instances": [1.0, 2.0, 5.0]}' -X POST http://localhost:8501/v1/models/half_plus_two:predict

import json, requests, sys, pickle, heapq
import tensorflow.contrib.keras as kr
from service_config import sentence_class_ip
reload(sys)
sys.setdefaultencoding('utf8')

#URL = 'http://localhost:8501/v1/models/sentence_class:predict'
#URL = 'http://192.168.7.218:8501/v1/models/sentence_class:predict'
#URL = 'http://192.168.7.205:8511/v1/models/sentence_class:predict'
URL = 'http://' + sentence_class_ip + '/v1/models/sentence_class:predict'

try:
    sentence = sys.argv[1]
except:
    sentence = '找工作'

'''
test_input = [[0] * 195 + [5, 11, 30, 80]]
data = json.dumps({"signature_name": "serving_default", "instances": test_input})
print 'Data : ', data

headers = {"content-type": "application/json"}
json_response = requests.post('http://localhost:8501/v1/models/sentence_class:predict', data=data, headers=headers)
predictions = json.loads(json_response.text)['predictions']

print 'predictions : ', predictions
'''

class Predictor():
    def __init__(self):
        self.map_path = './model/id.maps'
        with open(self.map_path, "rb") as f:
            self.word_to_id, self.cat_to_id, self.seq_length, self.num_classes = pickle.load(f)
        self.id_to_cat = {v: k for k, v in self.cat_to_id.items()}
    def predict(self, txt, topk=1):
        #print json.dumps(self.word_to_id, ensure_ascii=False), json.dumps(self.cat_to_id, ensure_ascii=False), self.seq_length, self.num_classes
        data_id = [[self.word_to_id[x] for x in list(txt.decode('utf-8')) if x in self.word_to_id]] #; print data_id
        x_pad = kr.preprocessing.sequence.pad_sequences(data_id, self.seq_length).tolist()   #; print x_pad;exit()
        data = json.dumps({"signature_name": "serving_default", "instances": x_pad})
        headers = {"content-type": "application/json"}
        json_response = requests.post(URL, data=data, headers=headers)
        predictions = json.loads(json_response.text)['predictions'] #;   print 'predictions : ', predictions
        pred_prob = predictions[0]  #; print pred_prob
        k_index = list(map(pred_prob.index, heapq.nlargest(topk, pred_prob)))
        k_value = heapq.nlargest(topk, pred_prob)
        res = []
        for i in range(len(k_index)):
            res.append((self.id_to_cat[k_index[i]], round(k_value[i], 3)))
        return res#[0]   # json.dumps({'sentence': txt, 'predict_result': res}, ensure_ascii=False)     # res[0]

if __name__ == '__main__':
    p = Predictor()
    print json.dumps(p.predict(sentence), ensure_ascii=False)

