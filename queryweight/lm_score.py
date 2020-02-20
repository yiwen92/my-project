from config import conf
from seg_utils import Tokenizer
import kenlm

class LanguageModelScore:
    """Value = {perplexity (entire query) / perplexity (entire query without current term)}. This feature reflects the query quality with/without current term"""
    def __init__(self):
        self.seg = Tokenizer()
        self.lm = kenlm.Model(conf.lm)
        self.punctuation_list = ".。,，,、?？:：;；{}[]【】“‘’”《》/!！%……（）<>@#$~^￥%&*\"\'=+-_——「」"
        self.stopwords = [e.strip() for e in open(conf.stop_words, encoding="utf8").readlines()]

    def weight_lm(self, sentence):
        senten2term, word_seg = self.seg.tokenize(sentence)
        total_score = self.lm.perplexity(' '.join(senten2term))
        weight, weight_sum = [], 0.0
        for i in range(len(senten2term)):
            tmp = [senten2term[j] for j in range(len(senten2term)) if i != j]
            score = self.lm.perplexity((' '.join(tmp)))
            val = total_score / score
            if senten2term[i] in self.punctuation_list or senten2term[i] in self.stopwords: val = 0.0
            weight.append((senten2term[i], val))
            weight_sum += val
        token_weight = [(k, round(v/weight_sum, 3)) for k, v in weight]
        return token_weight

if __name__ == "__main__":
    query = "java开发工程师"
    lms = LanguageModelScore()
    lms.weight_lm(query)