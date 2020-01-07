from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging, json, os, io, re, datetime, jieba
from sklearn.feature_extraction.text import CountVectorizer
from embedding_intent_classifier import EmbeddingIntentClassifier
from count_vectors_featurizer import CountVectorsFeaturizer

comment_regex = re.compile(r'<!--[\s\S]*?--!*>', re.MULTILINE)
section_regexes = {}
section_regexes['intent'] = re.compile(r'##\s*{}:(.+)'.format('intent'))
item_regex = re.compile(r'\s*[-\*+]\s*(.+)')

log_file = './train.log'
#if os.path.exists(log_file): os.remove(log_file)

def init_log():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=log_file,
                        filemode='a')

class TrainData():
    def __init__(self):
        self.examples = []
        self.intents = set()
        self.id = 0
    def add_example(self, intent, data):
        self.examples.append({'intent': intent, 'data': data})

    def load_data(self, resource_name):
        qes2ids = {}
        qes2labels = {}
        files = list_files(resource_name)
        data_sets = [_load(f) for f in files]
        data_sets = [ds for ds in data_sets if ds]
        if len(data_sets) == 1:
            training_data = data_sets[0]
        else:
            training_data = data_sets[0].merge(*data_sets[1:])
        for k, v in training_data.items():
            self.intents.add(k)
            for e in v:
                if e not in qes2ids: qes2ids[e] = []
                if e not in qes2labels: qes2labels[e] = []
                qes2ids[e].append(self.id)
                qes2labels[e].append(k)
                self.add_example(k, e)
                self.id += 1
        for qe, lb in qes2labels.items():
            if len(lb) < 2: continue
            for v in qes2ids[qe]:
                self.examples[v]['multi_intents'] = lb
        logging.info("Training data stats: \t- intent examples: {} ({} distinct intents)".format(len(self.examples), len(self.intents)))
        print("Training data stats: \t- intent examples: {} ({} distinct intents)".format(len(self.examples), len(self.intents)))

def list_directory(path):
    if not isinstance(path, str):
        raise ValueError("Resourcename must be a string type")

    if os.path.isfile(path):
        return [path]
    elif os.path.isdir(path):
        results = []
        for base, dirs, files in os.walk(path):
            # remove hidden files
            goodfiles = filter(lambda x: not x.startswith('.'), files)
            results.extend(os.path.join(base, f) for f in goodfiles)
        return results
    else:
        raise ValueError("Could not locate the resource '{}'."
                         "".format(os.path.abspath(path)))

def list_files(path):
    return [fn for fn in list_directory(path) if os.path.isfile(fn)]

def read_file(filename, encoding="utf-8-sig"):
    """Read text from a file."""
    with io.open(filename, encoding=encoding) as f:
        return f.read()

def _strip_comments(text):
    return re.sub(comment_regex, '', text)

def _find_section_header(line):
    for name, regex in section_regexes.items():
        match = re.search(regex, line)
        if match is not None:
            return name, match.group(1)
    return None

def _load(filename, fformat='md'):
    train_data = {}
    logging.info("Training data format of {} is {}".format(filename, fformat))
    print("Training data format of {} is {}".format(filename, fformat))
    txt = read_file(filename)
    s = _strip_comments(txt)
    for line in s.splitlines():
        line = line.strip()
        header = _find_section_header(line)
        if header:
            label = header[1]
            if label not in train_data:
                train_data[label] = []
        else:
            match = re.match(item_regex, line)
            if match:
                item = match.group(1)
                train_data[label].append(item)
    return train_data

class message:
    def __init__(self, text):
        self.text = ' '.join(list(jieba.cut(text)))     # text
        self.intent = None
        self.text_features = None
        self.intent_ranking = None
    def get_info(self):
        self.info = {}
        self.info['intent'] = self.intent
        self.info['intent_ranking'] = self.intent_ranking
        self.info['text'] = self.text
        return self.info

class Interpreter:
    def __init__(self):
        self.count_vectors_featurizer = None
        self.embedding_intent_classifier = None
    def load(self, model_path):
        self.cvf = CountVectorsFeaturizer()
        self.eic = EmbeddingIntentClassifier()
        self.cvf.load(model_path)
        self.eic.load(model_path)
    def parse(self, text):
        m = message(text)
        self.cvf.process(m)
        self.eic.process(m)
        return m.get_info()

def do_train(data, model):
    td = TrainData()
    td.load_data(data)
    #logging.info('intent_labels :', td.intents)
    logging.info('intent_labels : %s', json.dumps(list(td.intents), ensure_ascii=False))
    print('intent_labels : ', json.dumps(list(td.intents), ensure_ascii=False))

    cvf = CountVectorsFeaturizer()
    eic = EmbeddingIntentClassifier()

    cvf.train(td.examples)      # 得到离散矢量特征模型
    eic.train(td.examples)      # 训练意图分类模型
    model_path = os.getcwd() + '/' + model + '/model_' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    logging.info('Model saved to %s', model_path)
    os.makedirs(model_path)
    cvf.persist(model_path)     # 保存离散矢量模型
    eic.persist(model_path)     # 保存意图分类模型

if __name__ == '__main__':
    data_file = './data/sen_class_corp1.md'
    model_file = 'models'
    #'''
    init_log()
    logging.info('Start Train Model ...')
    do_train(data_file, model_file)
    logging.info('Train Model Finished\n')
    exit()
    #'''
    lists = os.listdir(model_file); lists.sort(key=lambda fn: os.path.getmtime(model_file + "/" + fn)); lists.remove('fast_model')
    model_path = './models/' + lists[-1]
    itr = Interpreter()
    itr.load(model_path)
    question = "怎么找工作"
    print(json.dumps(itr.parse(question), ensure_ascii=False))
    a=1