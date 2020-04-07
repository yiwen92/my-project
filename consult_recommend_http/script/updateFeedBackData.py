#coding:utf8
import logging
import MySQLdb
import traceback, json
import sys
from get_cv_content import resolve_cv
sys.path.append('../')

#env = 'dev'
env = 'test'
#env = 'online'
db_param_r = {};    db_param_w = {}

if env == 'test':
    db_param_r['host'] = "10.9.10.29"; db_param_r['user'] = "nlpuser"; db_param_r['passwd'] = "Nlp@123!@#"; db_param_r['db'] = "dashixiong";
    db_param_w['host'] = "10.9.10.29"; db_param_w['user'] = "nlpuser"; db_param_w['passwd'] = "Nlp@123!@#"; db_param_w['db'] = "dashixiong_algorithm";
elif env == 'online':
    db_param_r['host'] = "192.168.9.51"; db_param_r['user'] = "nlpuser"; db_param_r['passwd'] = "Nlp@123!@#"; db_param_r['db'] = "dashixiong";
    db_param_w['host'] = "192.168.9.51"; db_param_w['user'] = "nlpuser"; db_param_w['passwd'] = "Nlp@123!@#"; db_param_w['db'] = "dashixiong_algorithm";

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

class ConsultantFeedBack:
    def __init__(self, _id):
        self.id = _id
        self.record = {}
        self.record['feed_backs'] = {}
        self.record['resume'] = {}

    def addFeedBack(self, orientation, satisfaction, vitality):
        feed_back = {
                'orientation': orientation,
                'satisfaction': satisfaction,
                'vitality': vitality
                }
        self.record['feed_backs'][orientation] = feed_back

    def addResumeInfo(self, info_key, info_value):
        self.record['resume'][info_key] = info_value

def get_id_phone():
    id2phone = {}
    conn = MySQLdb.connect(
            host = db_param_r['host'],   #"10.9.10.29",
            user = db_param_r['user'],   #"nlpuser",
            passwd = db_param_r['passwd'],   #"Nlp@123!@#",
            port = 3306,
            db = db_param_r['db'],  #"dashixiong",
            charset = "utf8")
    conn.autocommit(1)
    cursor = conn.cursor()
    sql = 'select uid,phone from user_login'
    cursor.execute(sql)
    data_read = cursor.fetchall()
    for ele in data_read:
        uid = ele[0]; phone = str(ele[1])
        if phone.strip() == '': continue
        id2phone[uid] = phone
        #print uid, phone    #type(uid), type(phone); exit()
    return id2phone

def updateSQL(user_id, content = '', query = False, update = False, delete = False):
    conn = None
    try:
        '''
        conn = MySQLdb.connect(             # 开发机数据库
            host = "192.168.1.201",
            user = "devuser",
            passwd = "devuser",
            port = 3310,
            db = "dashixiong_algorithm",
            charset = "utf8"
            )
        '''
        conn = MySQLdb.connect(             # 测试机数据库
            host = db_param_w['host'],      #"10.9.10.29",
            user = db_param_w['user'],      #"nlpuser",
            passwd = db_param_w['passwd'],  #"Nlp@123!@#",
            port = 3306,
            db = db_param_w['db'],          #"dashixiong_algorithm",
            charset = "utf8"
            )

        db_table = 'user_algorithm'
        conn.autocommit(1)
        cursor = conn.cursor()

        #cursor.execute('select * from ' + db_table); print 'test read data\n', cursor.fetchall(); exit()        # TEST

        querySQL = "select * from " + db_table +" where user_id = " + str(user_id)
        increaseSQL = "insert into " + db_table + " (`user_id`,`content`) values (" + str(user_id) + ",'" + str(content) + "')"
        upSQL = "update " + db_table + " set content = '" + str(content) + "' where user_id = " + str(user_id)
        delSql = "delete from " + db_table + " where user_id = " + str(user_id)

        hasRecord = False
        cursor.execute(querySQL)
        data_read = cursor.fetchall()
        if len(data_read):
            hasRecord = True

        #print hasRecord, data_read; exit()

        if query:       # 查询数据记录
            return data_read
        elif update:    # 更新数据记录
            if hasRecord:   # 有记录则更新数据
                try:
                    cursor.execute(upSQL)
                    return True
                except:
                    conn.rollback()
                    return False
            else:           # 无记录则增加数据
                try:
                    cursor.execute(increaseSQL)
                    return True
                except:
                    conn.rollback()
                    return False
        elif delete:    # 删除数据记录
            if hasRecord:
                try:
                    cursor.execute(delSql)
                    return True
                except:
                    conn.rollback()
                    return False
        return False
    except Exception, e:
        logging.warn('utils_updateSQL_error=%s' % (str(repr(e))))
        tb = traceback.format_exc(); print 'traceback:%s' % str(tb)
        return None
    finally:
        if conn:
            conn.close()

def updateFeedBack():
    id2phone = get_id_phone()   #; print id2phone; exit()
    for k, v in id2phone.items():
        res = resolve_cv(v)     # 根据手机号码 v 获取简历字段 res
        if len(res) == 0: continue
        record = updateSQL(k, query = True) #; print record; exit()         # 查询数据库中id为 k 的记录 record
        if record:      # 数据库中有记录则更新
            record_id = record[0][0]
            if str(k) != str(record_id): continue
            record_content = record[0][1]
            try: content_dict = json.loads(record_content)
            except: content_dict = {}
            resume = getter_info(content_dict, 'resume', None, dict = True)
            feed_backs = getter_info(content_dict, 'feed_backs', None, dict = True)
            if resume is None: resume = {}
            if feed_backs is None: feed_backs = {}
        else:          # 数据库中无记录则新增
            resume = {}
            feed_backs = {}

        #print json.dumps(res, ensure_ascii=False), json.dumps(resume, ensure_ascii=False), json.dumps(feed_backs, ensure_ascii=False)
        if 'cv_industry' in res: resume['cv_industry'] = res['cv_industry']
        if 'cv_discipline' in res: resume['cv_discipline'] = res['cv_discipline']
        if 'cv_school' in res: resume['cv_school'] = res['cv_school']
        if 'cv_function' in res: resume['cv_function'] = res['cv_function']
        #print json.dumps(res, ensure_ascii=False), json.dumps(resume, ensure_ascii=False), json.dumps(feed_backs, ensure_ascii=False)
        tmp = {}
        tmp['resume'] = resume
        tmp['feed_backs'] = feed_backs
        update_str = json.dumps(tmp, ensure_ascii=False)
        #print 'update_str', update_str
        updateSQL(k, update_str, update = True)    # 更新数据库

def get_ids():
    id2phone = get_id_phone()   #; print id2phone
    for k, v in id2phone.items():
        print k

if __name__ == '__main__':
    get_ids(); exit()
    #print db_param_r, get_id_phone(); exit()
    #print json.dumps(resolve_cv(1), ensure_ascii=False); exit()
    updateFeedBack(); exit()

    cfb = ConsultantFeedBack(4)
    cfb.addFeedBack(1,2,3)
    cfb.addFeedBack(4,5,6)
    cfb.addResumeInfo('cv_function', '4038975:0.93,3000693:0.63')
    cfb.addResumeInfo('cv_industry', '27,28,414,3,2,5')
    cfb.addResumeInfo('cv_school', '102269')
    cfb.addResumeInfo('cv_discipline', '1130310')
    cfb_str = json.dumps(cfb.record, ensure_ascii=False)
    #print cfb.id, cfb_str; exit()
    strSQL = '{"22": [{"satisfaction": 50, "orientation": 1, "vitality": 3}, {"satisfaction": 8, "orientation": 2, "vitality": 5}, {"cv_function": "4038975:0.93,3000693:0.63"}, {"cv_industry": "27,28,414,3,2,5"}, {"cv_school": "102269"}, {"cv_discipline": "1130310"}]}'
    #print strSQL; exit()
    print cfb.id, cfb.record, updateSQL(cfb.id, cfb_str, query = False, update = True, delete = False)  # id, content, query, update, delete

