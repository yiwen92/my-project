import  pymysql, math
from tqdm import tqdm

def export(db_params, table_name, offset, number):
    results = []
    sql = 'select * from ' + table_name + ' limit %s, %s' % (offset, number)
    connect = pymysql.connect(**db_params)
    cursor = connect.cursor()
    cursor.execute(sql)
    fields = cursor.description
    datas = cursor.fetchall()   ;   #print(fields, '\n', datas)
    field_ids = [1, 2, 3, 6, 8]
    """
    (1)user_id: 用户ID, (2)jd_id: 职位ID, (3)resume_id: 简历ID
    (6)status: 状态: 1-投递成功, 2-被查看 ,3-筛选通过, 4-面试通知, 5-简历不合适, 6-简历被转发, 99-取消应聘
    (8)type: 0-未知, 1-主动投递, 2-高意向, 3-逸橙推荐, 4-内推
    """
    results.append('\t'.join([str(fields[col][0]) for col in field_ids]) + '\n')    # 记录字段的类型
    for row in range(len(datas)):  # 记录数据
        results.append('\t'.join([str(datas[row][col]) for col in field_ids]) + '\n')
    cursor.close()
    connect.close
    return results

def get_jdcv_ids():
    """
    /opt/app/mysql/bin/mysql -h192.168.8.141 -ukdd -pkd12934d -P3307    (8.52)
    通过主投的jd和cv行为数据得到标注样本集  (105938610)
    set names utf8;
    show full columns from automaticdelivery;
    """
    db_params = {
        "host": "192.168.8.141",
        "user": "kdd",
        "password": "kd12934d",
        "port": 3307,
        "db": "companyresume",
        "charset": "utf8"
    }
    total_record, num_record = 105938610, 1000000
    outputpath = 'jdcvids_' + str(total_record)
    ftd = open(outputpath, 'w')
    print("writing sample id data :%s" % (outputpath))
    for i in tqdm(range(math.ceil(total_record / num_record)), total=math.ceil(total_record / num_record)):
        number = num_record
        if i == int(total_record / num_record): number = total_record % num_record
        # print(i * num_record, '\t', number)
        res = export(db_params, 'automaticdelivery', i * num_record, number)
        if i == 0:
            ftd.write(''.join(res))
        else:
            ftd.write(''.join(res[1:]))
    ftd.close()

def get_search_label_data():
    results = []
    db_params = {
        "host": "192.168.8.218",
        "user": "kdd",
        "password": "kd12934d",
        "port": 3307,
        "db": "feedback",
        "charset": "utf8"
    }
    sql = "select * from cv_score_keyword_detail where task_id=2979 and is_correct > '-1'"
    connect = pymysql.connect(**db_params)
    cursor = connect.cursor()
    cursor.execute(sql)
    fields = cursor.description
    datas = cursor.fetchall()   #; print(fields, '\n\n', datas)
    results.append('\t'.join([str(e[0]) for e in fields]) + '\n')    # 记录字段的类型
    for data in datas:  # 记录数据
        results.append('\t'.join([str(e) for e in data]) + '\n')
    with open("feedback.res", "w", encoding="utf8") as fin:
        fin.write("".join(results))
    cursor.close()
    connect.close

if __name__ == '__main__':
    #get_jdcv_ids()
    get_search_label_data()
