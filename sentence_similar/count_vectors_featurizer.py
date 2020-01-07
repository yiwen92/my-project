#coding:utf8
from __future__ import division
from __future__ import unicode_literals
#from __future__ import print_function
from __future__ import absolute_import

import logging, json, traceback
import os
import re
from typing import Any, Dict, List, Optional, Text
import utils

logger = logging.getLogger(__name__)

class CountVectorsFeaturizer():
    """Bag of words featurizer

    Creates bag-of-words representation of intent features
    using sklearn's `CountVectorizer`.
    All tokens which consist only of digits (e.g. 123 and 99
    but not ab12d) will be represented by a single feature."""

    name = "intent_featurizer_count_vectors"

    provides = ["text_features"]

    requires = []

    defaults = {
        # the parameters are taken from
        # sklearn's CountVectorizer

        # regular expression for tokens
        "token_pattern": r'(?u)\b\w\w+\b',

        # remove accents during the preprocessing step
        "strip_accents": None,  # {'ascii', 'unicode', None}

        # list of stop words
        "stop_words": None,  # string {'english'}, list, or None (default)

        # min document frequency of a word to add to vocabulary
        # float - the parameter represents a proportion of documents
        # integer - absolute counts
        "min_df": 1,  # float in range [0.0, 1.0] or int

        # max document frequency of a word to add to vocabulary
        # float - the parameter represents a proportion of documents
        # integer - absolute counts
        "max_df": 1.0,  # float in range [0.0, 1.0] or int

        # set range of ngrams to be extracted
        "min_ngram": 1,  # int
        "max_ngram": 1,  # int

        # limit vocabulary size
        "max_features": None,  # int or None

        # if convert all characters to lowercase
        "lowercase": True,  # bool

        # handling Out-Of-Vacabulary (OOV) words
        # will be converted to lowercase if lowercase is True
        "OOV_token": None,  # string or None
        "OOV_words": []  # string or list of strings
    }

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["sklearn"]

    def _load_count_vect_params(self):
        # regular expression for tokens
        self.token_pattern = None   #self.component_config['token_pattern']

        # remove accents during the preprocessing step
        self.strip_accents = None   #self.component_config['strip_accents']

        # list of stop words
        self.stop_words = None  #self.component_config['stop_words']

        # min number of word occurancies in the document to add to vocabulary
        self.min_df = 1 #self.component_config['min_df']

        # max number (fraction if float) of word occurancies
        # in the document to add to vocabulary
        self.max_df = 1.0   #self.component_config['max_df']

        # set ngram range
        self.min_ngram = 1  #self.component_config['min_ngram']
        self.max_ngram = 1  #self.component_config['max_ngram']

        # limit vocabulary size
        self.max_features = None    #self.component_config['max_features']

        # if convert all characters to lowercase
        self.lowercase = True   #self.component_config['lowercase']

    # noinspection PyPep8Naming
    def _load_OOV_params(self):
        self.OOV_token = self.component_config['OOV_token']

        self.OOV_words = self.component_config['OOV_words']
        if self.OOV_words and not self.OOV_token:
            logger.error("The list OOV_words={} was given, but "
                         "OOV_token was not. OOV words are ignored."
                         "".format(self.OOV_words))
            self.OOV_words = []

        if self.lowercase and self.OOV_token:
            # convert to lowercase
            self.OOV_token = self.OOV_token.lower()
            if self.OOV_words:
                self.OOV_words = [w.lower() for w in self.OOV_words]

    def __init__(self, component_config=None):
        """Construct a new count vectorizer using the sklearn framework."""

        #super(CountVectorsFeaturizer, self).__init__(component_config)

        # parameters for sklearn's CountVectorizer
        self._load_count_vect_params()

        # handling Out-Of-Vacabulary (OOV) words
        #self._load_OOV_params()

        # declare class instance for CountVectorizer
        #self.vect = None

    def _tokenizer(self, text):
        """Override tokenizer in CountVectorizer"""
        text = re.sub(r'\b[0-9]+\b', '__NUMBER__', text)

        token_pattern = re.compile(self.token_pattern)
        tokens = token_pattern.findall(text)

        if self.OOV_token:
            if hasattr(self.vect, 'vocabulary_'):
                # CountVectorizer is trained, process for prediction
                if self.OOV_token in self.vect.vocabulary_:
                    tokens = [
                        t if t in self.vect.vocabulary_.keys()
                        else self.OOV_token for t in tokens
                    ]
            elif self.OOV_words:
                # CountVectorizer is not trained, process for train
                tokens = [
                    self.OOV_token if t in self.OOV_words else t
                    for t in tokens
                ]

        return tokens

    @staticmethod
    def _get_message_text(message):
        if message.get("spacy_doc"):  # if lemmatize is possible
            return ' '.join([t.lemma_ for t in message.get("spacy_doc")])
        elif message.get("tokens"):  # if directly tokens is provided
            return ' '.join([t.text for t in message.get("tokens")])
        else:
            return message.text

    def get_message_text(self, message):            # add by zouning on 2018/12/27
        return message.text.replace(' ', '')

    # noinspection PyPep8Naming
    def _check_OOV_present(self, examples):
        if self.OOV_token and not self.OOV_words:
            for t in examples:
                if self.OOV_token in t or (
                        self.lowercase and self.OOV_token in t.lower()):
                    return
            logger.warning("OOV_token='{}' was given, but it is not present "
                           "in the training data. All unseen words "
                           "will be ignored during prediction."
                           "".format(self.OOV_token))

    def train(self, training_data, cfg=None, **kwargs):
        # type: (TrainingData, RasaNLUModelConfig, **Any) -> None
        """Take parameters from config and
            construct a new count vectorizer using the sklearn framework."""
        import jieba
        from sklearn.feature_extraction.text import CountVectorizer

        spacy_nlp = kwargs.get("spacy_nlp")
        if spacy_nlp is not None:
            # create spacy lemma_ for OOV_words
            self.OOV_words = [t.lemma_
                              for w in self.OOV_words
                              for t in spacy_nlp(w)]
        '''
        self.vect = CountVectorizer(token_pattern=self.token_pattern,
                                    strip_accents=self.strip_accents,
                                    lowercase=self.lowercase,
                                    stop_words=self.stop_words,
                                    ngram_range=(1, 2),
                                    max_df=self.max_df,
                                    min_df=self.min_df,
                                    max_features=self.max_features,
                                    #analyzer='char',        # add by zouning on 2018/12/27
                                    tokenizer=self._tokenizer)
        '''

        '''
        lem_exs = [self._get_message_text(example)
                   for example in training_data.intent_examples]
        lem_exs = [self.get_message_text(example) for example in training_data.intent_examples]        # add by zouing on 2018/12/27
        self._check_OOV_present(lem_exs)
        '''

        X_test = [u'没有 你 的 地方 都是 他乡',u'没有 你 的 旅行 都是 流浪']
        count_vec = CountVectorizer(token_pattern=r"(?u)\b\w\w+\b", analyzer='char')
        a = count_vec.fit_transform(X_test).toarray()
        #lem_exs1 = ['我想吃饭火锅啊']
        #aa = count_vec.fit_transform(lem_exs1).toarray()

        lem_exs = []
        #lem_exs = [' '.join(list(jieba.cut(e['data']))) for e in training_data]
        for ele in training_data:  lem_exs.append(ele['data'].replace(' ', ''))
        self.vect = CountVectorizer(analyzer='char', ngram_range=(1, 2), min_df=2, max_df=5)
#        for ele in training_data:  lem_exs.append(ele['data'].replace(' ', ''))

        try:
            # noinspection PyPep8Naming
            X = self.vect.fit_transform(lem_exs).toarray()
        #except ValueError:
        except Exception as e:
            self.vect = None
            logging.error('Fail to generate count vector feature %s', str(e))
            return
        logging.info("Dictionary size: {}".format(len(self.vect.vocabulary_)))
        for i, example in enumerate(training_data):
            example['text_features'] = X[i]

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None
        if self.vect is None:
            logger.error("There is no trained CountVectorizer: "
                         "component is either not trained or "
                         "didn't receive enough training data")
        else:
            message_text = message.text

            bag = self.vect.transform([message_text]).toarray().squeeze()
            message.text_features = bag

    def persist(self, model_dir):
        # type: (Text) -> Dict[Text, Any]
        """Persist this model into the passed directory.
        Returns the metadata necessary to load the model again."""

        featurizer_file = os.path.join(model_dir, self.name + ".pkl")
        utils.pycloud_pickle(featurizer_file, self)
        return {"featurizer_file": self.name + ".pkl"}

    @classmethod
    def load(self, model_dir=None):
        featurizer_file = os.path.join(model_dir, self.name + ".pkl")
        model = utils.pycloud_unpickle(featurizer_file)
        self.vect = model.vect

if __name__ == '__main__':
    import jieba
    from sklearn.feature_extraction.text import CountVectorizer
    X_test = ['简历 上 的 哪些 内容 才 是 HR 眼中 的 干货', '简历 上 SVM 的 具体 流程']
    X = [' '.join(list(jieba.cut(e))) for e in X_test]
    count_vec = CountVectorizer(analyzer='char_wb', ngram_range=(1, 2), min_df=2, max_df=10)
    a = count_vec.fit_transform(X).toarray()
    b = count_vec.vocabulary_
    aa=1