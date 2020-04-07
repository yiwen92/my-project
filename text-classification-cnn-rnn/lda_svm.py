import json
import numpy as np
import tensorflow as tf
from tensorflow.contrib.session_bundle import exporter
'''
# 加载数据，使用sklearn自带的fetch_20newsgroups数据集
from sklearn.datasets import fetch_20newsgroups
dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
n_samples=200
data_samples = dataset.data[:n_samples] #截取需要的量，n_samples=2000

# CountVectorizer统计词频
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib  #也可以选择pickle等保存模型，请随意
n_features=2500
#构建词汇统计向量并保存，仅运行首次
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=n_features, stop_words='english')
tf = tf_vectorizer.fit_transform(data_samples)      ;   a=tf[0]
joblib.dump(tf_vectorizer,'tf_Model.pkl',compress=3)

#==============================================================================
#得到存储的tf_vectorizer,节省预处理时间
#from sklearn.externals import joblib
#tf_vectorizer = joblib.load('tf_Model.pkl')
#tf = tf_vectorizer.fit_transform(data_samples)
#==============================================================================

from sklearn.decomposition import LatentDirichletAllocation
n_topic = 10
n_topics = 30
lda = LatentDirichletAllocation(n_topics=n_topic, max_iter=50, learning_method='batch')
lda.fit(tf) #tf即为Document_word Sparse Matrix

def print_top_words(model, feature_names, n_top_words):
    #打印每个主题下权重较高的term
    for topic_idx, topic in enumerate(model.components_):
        print ("Topic #%d:" % topic_idx)
        print (" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
    #打印主题-词语分布矩阵
    print (model.components_)

n_top_words=20
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)

#print trained topic model
tf_feature_names = tf_vectorizer.get_feature_names()
for idx, topic in enumerate(lda.components_, start=1):
    print('Topic #%d' % idx)
    print("/".join([tf_feature_names[i] for i in topic.argsort ()[:-11:-1]]))   #打印（主题-词汇）向量

aaa=lda.transform(tf)[0]  #打印（文章-主题）向量

def lda_svm():
    data_samples = [e.strip().split('\t')[-1] for e in open('./data/label_qes1', 'r', encoding='utf8').readlines()]
    stop_words = [e.strip() for e in open('./data/stop_words.txt', 'r', encoding='utf8').readlines()]
    tf_vectorizer = CountVectorizer(stop_words=stop_words, analyzer='char')
    tf = tf_vectorizer.fit_transform(data_samples)    ;  aa=tf[0]
    lda = LatentDirichletAllocation(n_topics=10, max_iter=50, learning_method='batch')
    X_train = lda.fit(tf).transform(tf)
    tf_feature_names = tf_vectorizer.get_feature_names()
    print_top_words(lda, tf_feature_names, 10)
    a=1 ;   json.dumps()
'''
def spark_test():
    import pyspark
    from pyspark import SparkContext as sc
    from pyspark import SparkConf
    # conf=SparkConf().setAppName("miniProject").setMaster("local[*]")
    conf = SparkConf().setAppName("lg").setMaster("spark://192.168.9.140:8020")
    sc = sc.getOrCreate(conf)

    # （a）利用list创建一个RDD;使用sc.parallelize可以把Python list，NumPy array或者Pandas Series,Pandas DataFrame转成Spark RDD。
    rdd = sc.parallelize(list(range(10 ** 2)))
    # print(rdd)
    # Output:ParallelCollectionRDD[0] at parallelize at PythonRDD.scala:480

    # （b）getNumPartitions()方法查看list被分成了几部分
    print(rdd.getNumPartitions())

def tensorflow_serving():
    # Generate input data
    n_samples = 1000; learning_rate = 0.01; batch_size = 100; n_steps = 500
    x_data = np.arange(100, step=.1);    y_data = x_data + 20 * np.sin(x_data / 10)
    x_data = np.reshape(x_data, (n_samples, 1));    y_data = np.reshape(y_data, (n_samples, 1))
    # Placeholders for batched input
    x = tf.placeholder(tf.float32, shape=(batch_size, 1));    y = tf.placeholder(tf.float32, shape=(batch_size, 1))
    with tf.variable_scope('test'):
        w = tf.get_variable('weights', (1, 1), initializer=tf.random_normal_initializer())
        b = tf.get_variable('bias', (1,), initializer=tf.constant_initializer(0))
        y_pred = tf.matmul(x, w) + b
        loss = tf.reduce_sum((y - y_pred) ** 2 / n_samples)
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
        with tf.Session() as sess:
            saver = tf.train.Saver()
            sess.run(tf.initialize_all_variables())
            for _ in range(n_steps):
                indices = np.random.choice(n_samples, batch_size)
                x_batch = x_data[indices]
                y_batch = y_data[indices]
                _, loss_val = sess.run([opt, loss], feed_dict={x: x_batch, y: y_batch})
            print(w.eval()); print(b.eval()); print(loss_val)
            saver.save(sess, "./model/test.ckpt")   #;   saver.restore(sess, “./model/test.ckpt”)
            model_exporter = exporter.Exporter(saver)
            model_exporter.init(
                sess.graph.as_graph_def(),
                named_graph_signatures={
                    'inputs': exporter.generic_signature({'x': x}),
                    'outputs': exporter.generic_signature({'y': y_pred})})
            model_exporter.export('./model', tf.constant(1), sess)

    a=1

if __name__ == '__main__':
    #lda_svm()
    spark_test()
    #tensorflow_serving()
    pass