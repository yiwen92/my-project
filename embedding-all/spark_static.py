import os, traceback, logging, time, re, sys
from pyspark import SparkContext, SparkConf
from seg_utils import Tokenizer

static_jd_title = "titledesc"       # titledesc, title, desc

token = Tokenizer()

def parse_line_jdtitle(line):
    title = []
    try:
        seg_line = line.strip().split('\t')
        if seg_line[0].isdigit() and len(seg_line) >= 4:
            title.append("_".join([seg_line[0], seg_line[3]]))
    except Exception as e:
        logging.warning('parse_line_jdtitle_err=%s,line:%s' % (repr(e), line))
    return title

def parse_line_jd(line):
    desc = []
    try:
        seg_line = line.strip().split('\t')
        if seg_line[0].isdigit() and len(seg_line) >= 34:
            important_tokens = token.select_important_tokens(seg_line[33].replace('\\n', ""))
            desc = ["_".join([seg_line[0], e]) for e in important_tokens]
    except Exception as e:
        logging.warning('parse_line_jd_err=%s,line:%s' % (repr(e), line))
    return desc

def parse_line_jdtitledesc(line):
    res = []
    try:
        seg_line = line.strip().split('\t')
        tmp = [e.replace('\t', '').replace('\\n', '') for e in seg_line if e]
        if len(tmp) > 1:
            desc_req = " ".join([e for e in tmp[1:]])
            desc_req_imp = token.select_important_tokens(desc_req)
            title_desc = [tmp[0]] + desc_req_imp
            if len(title_desc) >= 2:
                res = ["\t".join(title_desc)]
    except Exception as e:
        logging.warning('parse_line_jdtitledesc_err=%s,line:%s' % (repr(e), line))
    return res

if static_jd_title == "titledesc":
    parse_function = parse_line_jdtitledesc
    hadoop_input_file = 'hdfs:///basic_data/position_name_desc_re/*'           # jd 标题和描述信息
    #hadoop_input_file = 'hdfs:///basic_data/position_name_desc_re/000115_0'
    output_file = 'hdfs:///user/kdd_zouning/jdtitledesc'
elif static_jd_title == "title":
    parse_function = parse_line_jdtitle
    hadoop_input_file = 'hdfs:///basic_data/jd/positions/20190301/*'           # jd 标题
    hadoop_input_file = 'hdfs:///basic_data/jd/positions/20190301/position_0/data__133f4412_7fa6_4ee5_8a26_907b9926d7f6'
    output_file = 'hdfs:///user/kdd_zouning/jdtitle'
else:
    parse_function = parse_line_jd
    hadoop_input_file = 'hdfs:///basic_data/jd/positions_extras/20190301/*'    # jd 描述
    hadoop_input_file = 'hdfs:///basic_data/jd/positions_extras/20190301/position_0/data__092fe84d_2188_49af_a4e4_d954708c08ac'
    output_file = 'hdfs:///user/kdd_zouning/jddesc'

SPK_CONF = SparkConf()\
        .setAppName("zn-%s"%re.sub(r".*/", "", sys.argv[0]))\
        .set('spark.driver.memory', '20g')\
        .set('spark.executor.memory', '20g')\
        .set('spark.driver.maxResultSize', '100g')\
        .set("spark.hadoop.validateOutputSpecs", "false") \
        .set('spark.executor.extraJavaOptions','-XX:MaxDirectMemorySize=10g')

def static_text(input_file, outputfile):
    try:
        os.system("hadoop fs -rm -r " + outputfile + "_*")
        output_file = outputfile + time.strftime('_%Y_%m_%d_%H_%M_%S',time.localtime(time.time()))
        sc = SparkContext(conf=SPK_CONF)
        lines = sc.textFile(input_file)
        #a = lines.flatMap(parse_function).map(lambda word: (word, 1)).reduceByKey(lambda a, b: a + b).sortBy(lambda x: x[1], False).map(lambda x: "%s\t%d" % (x[0], x[1])).collect()
        lines.flatMap(parse_function).map(lambda word: (word, 1)).reduceByKey(lambda a, b: a + b).sortBy(lambda x: x[1], False).map(lambda x: "%s&%d" % (x[0], x[1])).saveAsTextFile(output_file)
        logging.info(lines.first().split())
        os.system("hadoop fs -get " + output_file + " ./")
    except Exception as e:
        tb = traceback.format_exc();  logging.error('traceback:%s' % str(tb))

def test():
    jd_title_desc = [parse_line_jdtitledesc(line) for line in open("jdtitledesc.10000", encoding="utf8").readlines()]
    if static_jd_title == "1": hadoop_input_file = 'file:///D:/Python Project/entity_similar/data/jd_title.100'
    else: hadoop_input_file = 'file:///D:/Python Project/entity_similar/data/jd_desc.100'
    jd_title, jd_desc = "data/jd_title.100", "data/jd_desc.100"
    #jdtitle = [parse_line_jdtitle(line) for line in open(jd_title, encoding="utf8").readlines()]
    #jddesc = [parse_line_jd(line) for line in open(jd_desc, encoding="utf8").readlines()]
    static_text("file:///opt/userhome/kdd_zouning/entity_similar/jdtitledesc.10000", "")
    #static_text(hadoop_input_file, 'hdfs:///user/kdd_zouning/jdtitle')
    a=1

if __name__ == '__main__':
    #test()
    static_text(hadoop_input_file, output_file)
