#coding:utf8
'''其他模块的综合处理代码块,与tag的逻辑分开'''
import os,re,logging
#term weight
from term_weight_wrapper import QueryTermWeight
#公司改写
from corp_rewrite_wrapper import  CorpRewriter
#职能扩展
from func_rewrite_wrapper import FuncRewriter
#行业扩展
from ind_rewrite_wrapper import IndRewriter
#hukai kuozhan
from query_expand_rewrite_wrapper import QueryExpandRewriter
from query_weight_rewrite_wrapper import QueryWeightRewriter
cur_dir = os.path.dirname(__file__)
class ThirdModuleProcess:
    def __init__(self):
        #query term weight
        self.qtw = QueryTermWeight()
        
        #company 改写
        self.corp_rewr = CorpRewriter()

        #职能扩展
        self.qexp = FuncRewriter()

        #行业扩展
        self.ind_rewr = IndRewriter()

        #2020-03-11,add hukai and zouning client
        self.query_expand_rewr = QueryExpandRewriter()
        self.query_weighting_client = QueryWeightRewriter()

        logging.info('Init ThirdModuleProcess Done!')


    def process(self,tob_result,query_tags,ext_dict):
        '''---------------改写逻辑------------------'''
        tob_result['query_rewrite'] = {}
        tob_result['query_term_weight'] = {}
        tob_result['qtw_feed'] = ''
        #追加query term weight结果
        if ext_dict['switch_flag']['term_weight'] == '1':
            query_tw,feed_term = self.qtw.get_term_weight(query_tags)
            tob_result['query_term_weight'] = query_tw
            tob_result['qtw_feed'] = feed_term


        #追加公司改写的结果
        #增加开关控制是否进行公司改写
        if ext_dict['switch_flag']['rewrite/corp'] == '1':
            corp_rewrite_result = self.corp_rewr.rewrite(query_tags,ext_dict['rewrite/corp'])
            tob_result['query_rewrite']['company'] = corp_rewrite_result

        #职能扩展
        if ext_dict['switch_flag']['rewrite/func'] == '1':
            func_expansion_result = self.qexp.rewrite(query_tags,ext_dict['rewrite/func'])
            tob_result['query_rewrite']['function'] = func_expansion_result

        #行业扩展
        if ext_dict['switch_flag']['rewrite/ind'] == '1':
            ind_expansion_result = self.ind_rewr.rewrite(query_tags,ext_dict['rewrite/func'])
            tob_result['query_rewrite']['industry'] = ind_expansion_result
        
        #2020-03-10 add query expand from hukai and query weighting from zouning
        if ext_dict['switch_flag']['rewrite/query_expand'] == '1':
            query_expand_result = self.query_expand_rewr.expand(tob_result['raw_query'])
            tob_result['query_expand'] = query_expand_result
        if ext_dict['switch_flag']['rewrite/query_weight'] == '1':
            query_weighting_result = self.query_weighting_client.weight(tob_result['raw_query'])
            tob_result['query_weight'] = query_weighting_result


        return tob_result


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    tmpr = ThirdModuleProcess()
