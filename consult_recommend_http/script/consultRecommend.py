#!/usr/bin/env
#coding:utf8
from __future__ import division
import sys,json,time,math,datetime,csv
import logging
import utils
import traceback
from read_es import ElasticObj, query, queryMap
from get_company_score import companyScore
from service_config import es_ip
from get_cv_content import get_cv_info_by_uid
reload(sys)
sys.setdefaultencoding('utf8')

def init():
    logging.basicConfig(level=logging.DEBUG,
            format='%(asctime)s %(levelname)s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            filename='',
            filemode='a')

def getter_info(base_dict, base_key, default, dict = False):
    if base_key in base_dict:
        ret = str(base_dict[base_key])
        if ret.strip() == '':
            return default
        else:
            if dict:
                return base_dict[base_key]
            else:
                return ret
    else:
        return default

class consultant:
    def __init__(self, consultant_id = 0, match_score = 0.0, detail = {}):
        self.consultant_id = consultant_id
        self.match_score = match_score
        self.detail = detail
        self.consult_feedbacks = []

    def get_info(self):
        ret = {}
        ret['consultant_id'] = self.consultant_id
        ret['match_score'] = self.match_score
        ret['detail'] = self.detail
        return ret

    def create_consult_feedback(self, orientation, vitality, satisfaction):
        self.consult_feedbacks.append({'orientation': orientation, 'vitality': vitality, 'satisfaction': satisfaction})

    def get_consult_feedbacks(self):
        return {self.consultant_id: self.consult_feedbacks}

class ConsultRecommend:
    def __init__(self):
        self.VERSION = 'consult_recommend_20181129'
        self.es_ip = es_ip      # 开发：'192.168.1.107'；测试：'10.9.10.7'；线上：'192.168.8.190'
        self.es_port = 9200
        self.es_index_name = 'dashixiong'
        self.es_index_type = 'consultant'
        self.es_obj = ElasticObj(self.es_index_name, self.es_index_type, self.es_ip, self.es_port)
        self.cs = companyScore()
        self.func_threshold = 0.5
        self.match_score_direct = 5
        self.match_score_indirect = 2
        self.industry_match_score = 3
        self.pos_func_match_score = 4
        self.corp_match_score = 3
        self.corpo_score = 1
        self.feed_back_score = 0
        self.discipline_match_score = 1
        self.school_match_score = 1
        self.max_score = 1
        self.industry_dict = {line.strip().split(',')[0]: line.strip().split(',')[2] for line in list(open('./data/industry.csv', 'r').readlines())[1:]}
        self.corporation_dict = {line.strip().split(',')[0]: line.strip().split(',')[1] for line in list(open('./data/corporation.csv', 'r').readlines())[1:]}
        self.class2capacitypoint = {}
        self.read_class2capacity()
        logging.info('init consult recommend ok, version=[%s]' % self.VERSION)

    def expandId(self, id_list, id_dict):
        all_ids = id_list[:]
        for ele in id_list:
            cur = str(ele)
            while cur in id_dict and id_dict[cur] != '0':
                all_ids.append(int(cur))
                cur = id_dict[cur]
        return list(set(all_ids))

    def read_class2capacity(self):
        capacity = [line.strip().split(',') for line in open('./data/capacity_point.csv', 'r').readlines()]
        self.class2capacitypoint[capacity[0][0].strip()] = []
        self.class2capacitypoint[capacity[0][1]] = []
        self.class2capacitypoint[capacity[0][2]] = []
        self.class2capacitypoint[capacity[0][3]] = []   #; print json.dumps(self.class2capacitypoint, ensure_ascii=False); exit()
        for ele in capacity[1:]:
            if ele[0] != '0': self.class2capacitypoint[capacity[0][0]].append(ele[0])
            if ele[1] != '0': self.class2capacitypoint[capacity[0][1]].append(ele[1])
            if ele[2] != '0': self.class2capacitypoint[capacity[0][2]].append(ele[2])
            if ele[3] != '0': self.class2capacitypoint[capacity[0][3]].append(ele[3])
        #print 'capacity', json.dumps(self.class2capacitypoint, ensure_ascii=False); exit()

    def on_predict_begin(self):
        logging.debug('on_recommend_begin')
        self.logs = {}
        self.t_begin = time.time()

    def on_predict_end(self):
        logging.debug('on_recommend_end')
        phead = '[on_predict_end] | req_dict=%s | accurate_query=%s | accurate_match_number=%d | vague_match_number=%d | cost=%.3fs'
        logging.info(phead  %
                (self.logs['req_dict'], self.logs['accurate_query'], self.logs['accurate_match_number'], self.logs['vague_match_number'], (time.time() - self.t_begin)))

    def calHitScore(self, hit, baseScore = 2):
        #print 'hit', json.dumps(hit, ensure_ascii=False)#; exit()
        direct_match = 0.0; indirect_match = 0.0; industry_match = 0.0; pos_func_match = 0.0; corp_match = 0.0
        corp_score = 0.0; feed_back_score = 0.0; discipline_match = 0.0; school_match = 0.0
        detail = {}
        _id = getter_info(hit, '_id', None)
        _source = getter_info(hit, '_source', None, True)
        if _source:
            capacity_point = getter_info(_source, 'capacity_point', None, True)
            industry = getter_info(_source, 'industry', None, True)
            industry_id = getter_info(_source, 'industry_id', None, True)
            industry_id = self.expandId(industry_id, self.industry_dict)
            resume = getter_info(_source, 'resume', None, True) #; print 'hit', json.dumps(hit, ensure_ascii=False), 'resume', resume; exit()
            if resume:
                cv_function = getter_info(resume, 'cv_function', None, True)
                cv_industry = getter_info(resume, 'cv_industry', None, True)
                #print 'hit', json.dumps(hit, ensure_ascii=False), '\nresume', resume, 'cv_function', cv_function, 'cv_industry', cv_industry; exit()
            # 能力点匹配分数
            if capacity_point:
                for co in self.match_direct:
                    if co in capacity_point:
                        direct_match = self.match_score_direct
                for co in self.match_indirect:
                    if co in capacity_point:
                        indirect_match = self.match_score_indirect
            # 行业匹配分数
            indus_match = False
            if industry:
                for indus in industry:
                    if not indus_match and indus in self.industry:
                        industry_match = self.industry_match_score
                        indus_match = True
                for indusid in industry_id:
                    if not indus_match and indusid in self.industry_id:
                        industry_match = self.industry_match_score
                        indus_match = True
            if resume and cv_industry:
                for indusid in cv_industry:
                    if not indus_match and indusid in self.industry_id:
                        industry_match = self.industry_match_score
                        indus_match = True
            # 职位或职能匹配分数
            function_should = getter_info(_source, 'function_should', None, True)
            function_must = getter_info(_source, 'function_must', None, True)
            position = getter_info(_source, 'position', None, True)
            func_match = False
            if function_should:
                for func in function_should:
                    if not func_match and func.split('^')[0] in self.func_id and float(func.split('^')[-1]) > self.func_threshold:
                        pos_func_match = self.pos_func_match_score
                        func_match = True
            if function_must:
                for func in function_must:
                    if not func_match and func.split('^')[0] in self.func_id and float(func.split('^')[-1]) > self.func_threshold:
                        pos_func_match = self.pos_func_match_score
                        func_match = True
            if position:
                for pos in self.position_tag_predict:
                    if not func_match and pos.split(':')[1] in position:
                        pos_func_match = self.pos_func_match_score
                        func_match = True
            if resume and cv_function:
                for ele in cv_function:
                    if not func_match and ele.split(':')[0] in self.func_id:
                        pos_func_match = self.pos_func_match_score
                        func_match = True
            # 公司匹配分数
            company_match = False
            company = getter_info(_source, 'company', None, True)
            company_id = getter_info(_source, 'company_id', None, True)
            company_id = self.expandId(company_id, self.corporation_dict)
            if company:
                for com in company:
                    if not company_match and com in self.company:
                        corp_match = self.corp_match_score
                        company_match = True
            if company_id:
                for comid in company_id:
                    if not company_match and comid in self.company_id:
                        corp_match = self.corp_match_score
                        company_match = True
            # 咨询师本身公司的分数
            company_id = getter_info(_source, 'company_id', None, True)
            if company_id:
                for com_score in self.cs.get_company_score(company_id):
                    corp_score = round(com_score, 3)
            # 反馈分数
            consult_feedback = getter_info(_source, 'feedback', None, True)
            if consult_feedback:        # 前期不考虑反馈分数
                feed_back_score = 0
            # 专业匹配
            if resume:
                cv_discipline = getter_info(resume, 'cv_discipline', None, True)
                if cv_discipline:
                    for ele in self.discipline_id:
                        if ele in cv_discipline:
                            discipline_match = self.discipline_match_score
            # 学校匹配
                cv_school = getter_info(resume, 'cv_school', None, True)
                if cv_school:
                    for ele in self.school_id:
                        if ele in cv_school:
                            school_match = self.school_match_score

        consult_score = 1 * baseScore + 2 * direct_match + 0.2 * indirect_match + 0.8 * industry_match + 1.5 * pos_func_match + 0.6 * corp_match \
                        + 0.6 * corp_score + 0.8 * feed_back_score  + 0.2 * discipline_match + 0.2 * school_match
        self.max_score = 1 * baseScore + 2 * self.match_score_direct + 0.2 * self.match_score_indirect + 0.8 * self.industry_match_score \
        + 1.5 * self.pos_func_match_score + 0.6 * self.corp_match_score + 0.6 * self.corpo_score + 0.8 * self.feed_back_score \
        + 0.2 * self.discipline_match_score + 0.2 * self.school_match_score

        detail['direct_match'] = direct_match; detail['indirect_match'] = indirect_match; detail['industry_match'] = industry_match
        detail['pos_func_match'] = pos_func_match; detail['corp_match'] = corp_match; detail['corp_score'] = corp_score
        detail['feed_back'] = feed_back_score; detail['baseScore'] = baseScore
        detail['discipline_match'] = discipline_match; detail['school_match'] = school_match
        consult_score = 100 * (consult_score / self.max_score)

        match_number = 1;   match_number_all = 6
        if direct_match > 0: match_number += 1
        if indirect_match > 0: match_number += 1
        if industry_match > 0: match_number += 1
        if pos_func_match > 0: match_number += 1
        if corp_match > 0: match_number += 1
        #if corp_score > 0: match_number += 1
        #print match_number, match_number_all, consult_score, self.max_score, json.dumps(detail, ensure_ascii=False); exit()
        consult_score += (100 - consult_score) * (match_number / match_number_all)
        return consult_score, detail

    # predict model
    def consultant_predict(self, req_dict):
        logging.debug('enter_consultant_predict')
        t0 = time.time()
        results = []
        self.match_indirect = []
        self.match_direct = []
        self.position_tag_predict = []
        self.func_id = []
        self.company_id = []
        self.industry_id = []
        self.industry = []
        self.company = []
        self.adviser_blacklist = []
        self.discipline_id = []
        self.school_id = []
        try:
            self.logs['req_dict'] = json.dumps(req_dict, ensure_ascii=False)
            #print 'req_dict : ', json.dumps(req_dict, ensure_ascii=False)
            consult_score_detail = {}
            consult_orientation = getter_info(req_dict, 'consult_orientation', None)
            consult_question = getter_info(req_dict, 'consult_question', None)
            adviser_blacklist = getter_info(req_dict, 'adviser_blacklist', None)
            re_cnt = getter_info(req_dict, 'return_count', None)
            uid = getter_info(req_dict, 'uid', None)
            if uid:
                cv_info = get_cv_info_by_uid(uid)
                if 'cv_industry' in cv_info:
                    for e in cv_info['cv_industry'].split(','):
                        self.industry_id.append(int(e))
                if 'cv_function' in cv_info:
                    for e in cv_info['cv_function'].split(','):
                        self.func_id.append(e.split(':')[0])
                if 'cv_school' in cv_info: self.school_id.extend(cv_info['cv_school'].split(','))
                if 'cv_discipline' in cv_info: self.discipline_id.extend(cv_info['cv_discipline'].split(','))
            #print cv_info, '\n', self.industry_id, self.func_id, self.school_id, self.discipline_id; exit()
            if re_cnt:
                re_cnt = int(re_cnt)
            else:
                re_cnt = 10
            if adviser_blacklist:
                self.adviser_blacklist = adviser_blacklist.split(',')
            #print self.adviser_blacklist
            consult_orientation_list = []
            if consult_orientation:
                consult_orientation_list = consult_orientation.split(',')
            #logging.info('consult_orientation_list=[%s], consult_question=[%s]' % (json.dumps(consult_orientation_list, ensure_ascii=False), consult_question))
            q_accurate = query()
            # 根据问题分类和能力点构建检索query
            for co in consult_orientation_list:
                self.match_direct.append(co)
                q_accurate.queryBuilder(queryMap('capacity_point', co, 'should'))
                if co in self.class2capacitypoint:
                    for e in self.class2capacitypoint[co]:
                        #print json.dumps(e, ensure_ascii=False)
                        self.match_indirect.append(e)
                        q_accurate.queryBuilder(queryMap('capacity_point', e, 'should'))

            #q_accurate.queryBuilder(queryMap('id', 3, 'must'))          #TEST
            #print json.dumps(self.match_indirect, ensure_ascii=False);    exit()

            if consult_question:          # 问题文本
                nerEntity = utils.getNerRes(consult_question)       # 问题文本NER识别实体
                label_probs = utils.getSentenceLbel(consult_question)   # 句子标签识别模型
                if label_probs and label_probs[0][1] >= 0.5:
                    self.match_direct.append(label_probs[0][0])
                    q_accurate.queryBuilder(queryMap('capacity_point', label_probs[0][0], 'should'))
                #print json.dumps(label_probs, ensure_ascii=False), json.dumps(self.match_direct, ensure_ascii=False); exit()
                if nerEntity:
                    # 职位
                    if 'IMPORTANT_POSITION' in nerEntity:
                        positions = list(set(nerEntity['IMPORTANT_POSITION']))
                        for pos in positions:
                            for tag_pre in utils.getFunc(pos):      # 职位识别出职能及id权重
                                if float(tag_pre.split(':')[-1]) < self.func_threshold: continue
                                self.position_tag_predict.append(tag_pre)
                                self.func_id.append(tag_pre.split(':')[0])
                            q_accurate.queryBuilder(queryMap('position', pos, 'should'))    # 职位 query
                    # 行业
                    if 'IMPORTANT_INDUSTRY' in nerEntity:
                        industrys = list(set(nerEntity['IMPORTANT_INDUSTRY']))
                        for indus in industrys:
                            self.industry.append(indus)
                            q_accurate.queryBuilder(queryMap('industry', indus, 'should'))    # 行业 query
                    # 公司
                    if 'IMPORTANT_CORPORATION' in nerEntity:
                        corps = list(set(nerEntity['IMPORTANT_CORPORATION']))
                        for cor in corps:
                            corp_tag = utils.getCorp(cor)   # 公司识别出id和行业id
                            self.company.append(cor)
                            if 'company_id' in corp_tag and corp_tag['company_id'] != 0:    # 公司 id，去掉无效的id(0)
                                self.company_id.append(corp_tag['company_id'])
                            q_accurate.queryBuilder(queryMap('company', cor, 'should'))    # 公司 query
                            if 'first_trade_list' in corp_tag:      # 行业 id
                                for ft in corp_tag['first_trade_list']:
                                    self.industry_id.append(ft)
                    # 实体
                    if 'ENTITY' in nerEntity:
                        entities = list(set(nerEntity['ENTITY']))
                        for en in entities:
                            self.match_direct.append(en)
                            q_accurate.queryBuilder(queryMap('capacity_point', en, 'should'))
            #print self.func_id; exit()
            self.company_id = self.expandId(self.company_id, self.corporation_dict)
            self.industry_id = self.expandId(self.industry_id, self.industry_dict)
            for ptp in list(set(self.position_tag_predict)):
                q_accurate.queryBuilder(queryMap('position', ptp.split(':')[1], 'should'))    # 职位识别的tag对应的query
            for fun in list(set(self.func_id)):
                q_accurate.queryBuilder(queryMap('function_should', fun, 'should'))    # 职能id对应的query
                q_accurate.queryBuilder(queryMap('function_must', fun, 'should'))    # 职能id对应的query
            for coid in list(set(self.company_id)):
                q_accurate.queryBuilder(queryMap('company_id', coid, 'should'))    # 公司id query
            for inid in list(set(self.industry_id)):
                q_accurate.queryBuilder(queryMap('industry_id', inid, 'should'))    # 行业id query

            #print 'accurate query: ', json.dumps(q_accurate.query, ensure_ascii=False); exit()
            self.logs['accurate_query'] = json.dumps(q_accurate.query, ensure_ascii=False)

            if q_accurate.hasQuery:
                searched = self.es_obj.getDataByQuery(q_accurate)    # 精确匹配
            #print q_accurate.hasQuery, '精确匹配', json.dumps(searched, ensure_ascii=False); exit()
            if q_accurate.hasQuery and searched and searched['hits']['total']:
                for hit in searched['hits']['hits']:
                    _id = getter_info(hit, '_id', None)
                    if _id in consult_score_detail or _id in self.adviser_blacklist: continue      # 过滤已经在结果中和黑名单中的id
                    score, detail = self.calHitScore(hit, 2)
                    consult_score_detail[_id] = [score, detail]
            #print len(consult_score_detail), json.dumps(consult_score_detail, ensure_ascii=False)#; exit()
            self.logs['accurate_match_number'] = len(consult_score_detail)
            if len(consult_score_detail) < re_cnt:      # 模糊匹配
                searched = self.es_obj.getAllData(10*re_cnt)
                #print re_cnt, json.dumps(searched, ensure_ascii=False); exit()
                if searched and searched['hits']['total']:
                    for hit in searched['hits']['hits']:
                         _id = getter_info(hit, '_id', None)
                         if _id in consult_score_detail or _id in self.adviser_blacklist: continue
                         score, detail = self.calHitScore(hit, 1)
                         consult_score_detail[_id] = [score, detail]
            #logging.info('acc_query=[%s], vague_query=[%s]' % (json.dumps(q_accurate.query, ensure_ascii=False), json.dumps(q_vague.query, ensure_ascii=False)))
            #sech = self.es_obj.getAllData(); print sech['hits']['total'],  json.dumps(sech, ensure_ascii=False); exit()   # 检索全部的数据
            self.logs['vague_match_number'] = len(consult_score_detail) - self.logs['accurate_match_number']

            #print json.dumps(consult_score_detail, ensure_ascii=False); exit()
            if consult_score_detail:       # 对结果进行排序
                sorted_consult = sorted(consult_score_detail.iteritems(), key=lambda d:d[1][0], reverse = True)
                for ele in sorted_consult:
                    results.append(consultant(ele[0], round(ele[1][0], 3), ele[1][1]).get_info())
            return_cnt = min(len(results), re_cnt)
            results = results[0:return_cnt]      # 取前 return_cnt 结果返回
            #print 'consult_score_detail', json.dumps(consult_score_detail, ensure_ascii=False), '\nsorted_consult', json.dumps(sorted_consult, ensure_ascii=False)
            #print  len(results), json.dumps(results, ensure_ascii=False); exit()
            #print time.time() - t0
            return results
        except Exception, e:
            logging.warn('recommend failed, param_request=%s, err=%s, cost=%s' % ( str(req_dict), repr(e), (time.time() - t0)))
            tb = traceback.format_exc();  logging.error('traceback:%s' % str(tb))
            return None

    def run(self, req_dict):
        self.on_predict_begin()
        callback_response = {}
        try:
            sorted_results = []
            self.request = req_dict
            logging.debug('prepare to response')

            # assemble result
            time_response_out = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

            #logging.info('req_dict=[%s]' % json.dumps(req_dict, ensure_ascii=False))

            rets = self.consultant_predict(req_dict)
            if rets:
                callback_response['recommend_results'] = rets
            else:
                callback_response['recommend_results'] = []

            #print 'callback_response\n', json.dumps(callback_response, ensure_ascii=False)
        except Exception,e:
            logging.warn('on_recommend_error=%s' % (str(repr(e))))
        self.logs['callback_response'] = json.dumps(callback_response, ensure_ascii=False)
        self.on_predict_end()
        return callback_response

if __name__ == '__main__':
    #init()
    cr = ConsultRecommend();    #print json.dumps(cr.class2capacitypoint, ensure_ascii=False); exit()
    #cs = companyScore(); print cs.get_company_score([1250807, 1045278]); exit()
    #es_obj = ElasticObj(cr.es_index_name, cr.es_index_type, cr.es_ip, cr.es_port)
    #q = query(); q.queryBuilder(queryMap('capacity_point', '面试技巧', 'should')); print(json.dumps(es_obj.getDataByQuery(q), ensure_ascii=False)); exit()
    #st = "熟悉互联网行业机器学习、深度学习等相关技术 ，担任过产品经理职位，阿里巴巴，百度相关工作经验"
    qid = '求职应聘'
    st = "想要做一名产品经理，我需要去学哪些技能或者课程？"
    advblac = ''
    cnt = 2
    uid = 232
    req_dict={
            'header':{},
            'request':{
                'c':'',
                'm':'consult_recommend',
                'p':{
                    'consult_question': st,#'职业选择工程师',
                    'consult_orientation': qid,#'求职应聘'
                    'adviser_blacklist': advblac,#'1,2,3'
                    'count': cnt,
                    'uid': uid,
                    }
                }
            }
    #print json.dumps(req_dict, ensure_ascii=False); #exit()
    user_dict = {
            'consult_orientation': req_dict['request']['p']['consult_orientation'],
            'consult_question': req_dict['request']['p']['consult_question'],
            'adviser_blacklist': req_dict['request']['p']['adviser_blacklist'],
            'return_count': req_dict['request']['p']['count'],
            'uid': req_dict['request']['p']['uid']
            }
    t0 = time.time()
    #sech = cr.es_obj.getAllData(2); print sech['hits']['total'],  json.dumps(sech, ensure_ascii=False); exit() # 获取所有数据
    ret = cr.run(user_dict); #print cr.logs['accurate_match_number'], cr.logs['vague_match_number']  #json.dumps(cr.logs, ensure_ascii=False)
    #print cr.company_id, cr.expandId(cr.company_id, cr.corporation_dict),'\n', cr.industry_id, cr.expandId(cr.industry_id, cr.industry_dict); exit()
    t1 = time.time()
    print 'time cost ', t1 - t0, '\nrun result\n', cr.max_score, len(ret['recommend_results']), json.dumps(ret, ensure_ascii=False)

