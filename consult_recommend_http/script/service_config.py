#!/usr/bin/env
# coding:utf8

# 初始化变量
env_dev = False      # 开发环境
env_test = False      # 测试环境
env_online = False      # 线上环境

# 配置环境
#env_dev = True      # 开发环境
#env_test = True      # 测试环境
#env_online = True      # 线上环境

# 根据 uid 获取手机号码来获取简历信息
db_param_r = {}
db_param_r['user'] = "nlpuser"
db_param_r['passwd'] = "Nlp@123!@#"
db_param_r['db'] = "dashixiong"

if env_dev:       # 开发环境
    es_ip = '192.168.1.107'                         # Elasticsearch 读取咨询师索引服务
    corp_tag_ip = 'http://192.168.1.204:51699'      # 公司识别为公司id和行业id服务（corp_tag）
    func_tag_ip = 'http://dev.nlp.rpc'              # 职位识别为职能id、权重服务（tag_predict_for_edps）
    #ner_tag_ip = 'http://192.168.1.111:51631'       # 问题文本ner实体抽取服务（echeng_ner_server）
    ner_tag_ip = 'http://211.148.28.36:51631'       # ner 服务测试环境中转接口
    cv_ip = 'http://dev.icdc.rpc'                   # 根据 id 或 phone 获取简历信息
elif env_test:     # 测试环境
    es_ip = '10.9.10.7'                             # Elasticsearch 读取咨询师索引服务
    corp_tag_ip = 'http://10.9.10.10:51699'         # 公司识别为公司id和行业id服务（corp_tag）
    func_tag_ip = 'http://testing2.nlp.rpc'         # 职位识别为职能id、权重服务（tag_predict_for_edps）
    ner_tag_ip = 'http://10.9.10.23:51631'          # 问题文本ner实体抽取服务（echeng_ner_server）
    cv_ip = 'http://testing2.icdc.rpc'              # 根据 id 或 phone 获取简历信息
    db_param_r['host'] = "10.9.10.29"               # 获取手机号码
    sentence_class_ip = '10.9.10.23:8511'        # 句子标签识别
elif env_online:     # 线上环境
    es_ip = '192.168.8.190'                         # Elasticsearch 读取咨询师索引服务
    corp_tag_ip = 'http://algo.rpc'                 # 公司识别为公司id和行业id服务（corp_tag）
    func_tag_ip = 'http://nlp.rpc'                  # 职位识别为职能id、权重服务（tag_predict_for_edps）
    ner_tag_ip = 'http://algo.rpc'                  # 问题文本ner实体抽取服务（echeng_ner_server）
    cv_ip = 'http://icdc.rpc'                       # 根据 id 或 phone 获取简历信息
    db_param_r['host'] = "192.168.9.51"             # 获取手机号码
    sentence_class_ip = '192.168.7.205:8511'        # 句子标签识别
else:             # 默认为线上环境
    es_ip = '192.168.8.190'                         # Elasticsearch 读取咨询师索引服务
    corp_tag_ip = 'http://algo.rpc'                 # 公司识别为公司id和行业id服务（corp_tag）
    func_tag_ip = 'http://nlp.rpc'                  # 职位识别为职能id、权重服务（tag_predict_for_edps）
    ner_tag_ip = 'http://algo.rpc'                  # 问题文本ner实体抽取服务（echeng_ner_server）
    cv_ip = 'http://icdc.rpc'                       # 根据 id 或 phone 获取简历信息
    db_param_r['host'] = "192.168.9.51"             # 获取手机号码
    sentence_class_ip = '192.168.7.205:8511'        # 句子标签识别

