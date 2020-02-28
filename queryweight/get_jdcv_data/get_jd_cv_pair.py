#!/usr/bin/env
#coding:utf8
import MySQLdb
import time
import xlwt
import json

#reader
def db_connect(db_params):
    #establish connection
    #print db_params
    db_conn = MySQLdb.connect(
            host = db_params['db_host'],
            user = db_params['db_user'],
            passwd = db_params['db_passwd'],
            port = db_params['db_port'],
            db = db_params['db_name'],
            charset = db_params['db_charset'])
    db_conn.autocommit(1) # set autocommit
    db_cursor = db_conn.cursor()
    #logging.debug('connected to db: %s'%(str(db_params['db_name'])))
    #print 'connected to db: %s'%(str(db_params['db_name']))
    return {'conn':db_conn,'cursor':db_cursor}

def db_reader(db_cursor,cmd_db_reader,is_print):
    data_read = None
    try:
        t0 = time.time()
        db_cursor.execute(cmd_db_reader)
        data_read = db_cursor.fetchall()	# 搜取所有结果
        #logging.debug('read db data ok, cost %.3fs' % ( time.time() -t0))
	#print 'read db data ok, cost %.3fs' % ( time.time() -t0)
        if is_print:
            print data_read
            print len(data_read)
    except Exception, e:
	print '\ndb_reader_error=%s' % repr(e)
        return None
    return data_read

def export(db_params, table_name, outputpath, records = 1000):
    db_handler = db_connect(db_params)
    #cmd_sql = 'select * from automaticdelivery limit 5'
    cmd_sql = 'select * from ' + table_name + ' limit %s' % records
    #print 'cmd_sql',cmd_sql;exit()
    data = db_reader(db_handler['cursor'],cmd_sql,False)
    fields = db_handler['cursor'].description	# 获取MYSQL里面的数据字段名称
    #print 'fields',fields
    #print 'data',data;exit()
    '''
    workbook = xlwt.Workbook()
    sheet = workbook.add_sheet('table_'+table_name,cell_overwrite_ok=True)
    for field in range(0,len(fields)):		# 写上字段信息
        sheet.write(0,field,fields[field][0])
    row = 1		# 获取并写入数据段信息
    col = 0
    for row in range(1,len(data)+1):
        for col in range(0,len(fields)):
            sheet.write(row,col,u'%s'%data[row-1][col])
	    #print row,col,u'%s'%data[row-1][col]
    workbook.save(outputpath)
    '''
    field_ids = [1, 2, 3, 6, 8]
    """
    (1)user_id: 用户ID, (2)jd_id: 职位ID, (3)resume_id: 简历ID
    (6)status: 状态: 1-投递成功, 2-被查看 ,3-筛选通过, 4-面试通知, 5-简历不合适, 6-简历被转发, 99-取消应聘
    (8)type: 0-未知, 1-主动投递, 2-高意向, 3-逸橙推荐, 4-内推
    """
    ftd = open(outputpath, 'w')
    #print fields, '\n', data
    for col in field_ids:       # 写入字段的类型
	    ftd.write(str(fields[col][0]) + '\t')
    ftd.write('\n')
    for row in range(len(data)):  # 写入数据
        #print data[row]
        for col in field_ids:
	        ftd.write(str(data[row][col]) + '\t')
        ftd.write('\n')
    ftd.close()

def read_cv(db_params, db_name, table_name, cv_id):
    db_params['db_name'] = str(db_name)
    db_handler = db_connect(db_params)
    #cmd_sql = 'select * from ' + table_name + ' limit %s' % 1
    cmd_sql = 'select compress from ' + table_name + ' where id = %s' % cv_id
    #print cmd_sql
    data = db_reader(db_handler['cursor'],cmd_sql,False)
    resumes_extras = json.loads(data[0][0].decode('zlib'))
    basic_info = json.loads(data[0][0].decode('zlib'))['basic']
    work_dict_origin = json.loads(data[0][0].decode('zlib'))['work']
    #print data[0][0].decode('zlib')
    cmd_sql1 = 'select data from resumes_algorithms where id = %s' % cv_id
    algorithms = db_reader(db_handler['cursor'],cmd_sql1,False)
    #print 'algorithms', algorithms[0][0] #json.dumps(algorithms, ensure_ascii=False)

    #print 'basic_info\n',json.dumps(basic_info, ensure_ascii=False), '\nwork_dict_origin\n',json.dumps(work_dict_origin, ensure_ascii=False);exit()
    cv_info = {
            'resumes_extras':resumes_extras,
            'algorithms':json.loads(algorithms[0][0]),
            'basic':basic_info,         #
            'work':work_dict_origin     #
            }
    #print json.dumps(cv_info, ensure_ascii=False)
    return cv_info


def read_cv_(db_params, db_name, cv_id):
    db_params['db_name'] = str(db_name)
    db_handler = db_connect(db_params)
    #cmd_sql = 'select * from ' + 'resumes_extras'  + ' limit %s' % 1
    cmd_sql = 'select compress from resumes_extras where id = %s' % cv_id
    #print cmd_sql
    data = db_reader(db_handler['cursor'],cmd_sql,False)
    resumes_extras = json.loads(data[0][0].decode('zlib'))
    cmd_sql1 = 'select data from resumes_algorithms where id = %s' % cv_id
    algorithms = db_reader(db_handler['cursor'],cmd_sql1,False)
    #print 'algorithms', json.dumps(algorithms, ensure_ascii=False);exit()
    cv_info = {
            'resumes_extras':resumes_extras,
            'algorithms':algorithms
            }
    #print 'cv_info', json.dumps(cv_info, ensure_ascii=False);exit()
    return cv_info

if __name__ == '__main__':
    """
    /opt/app/mysql/bin/mysql -h192.168.8.141 -ukdd -pkd12934d -P3307    (8.52)
    通过主投的jd和cv行为数据得到标注样本集
    set names utf8;
    show full columns from automaticdelivery;
    """
    #'''
    db_params = {
        'db_host': '192.168.8.141',
        'db_user': 'kdd',
        'db_passwd': 'kd12934d',
        'db_port': 3307,
        'db_name': 'companyresume',
        'db_charset': 'utf8'
        }
    readRecords = 10000
    export(db_params, 'automaticdelivery', './jd_cv_IDs', readRecords)
    print 'Finish generate jd-cv pairs :%d' % readRecords
    '''
    db_params = {
            'db_host': '192.168.8.112',
            'db_user': 'kdd',
            'db_passwd': 'kd12934d',
            'db_port': 3306,
            'db_charset': 'utf8'
            }
    read_cv(db_params, 'toh_resumes_6', 'resumes_extras', 1501917806)
    '''
