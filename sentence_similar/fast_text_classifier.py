import fastText.FastText as ff

def fast_text_train(data_file, model_file, test_file):
    classifier = ff.train_supervised(data_file)
    classifier.save_model(model_file)  # 保存模型
    test = classifier.test(test_file, 1)  # 输出测试结果
    b=test.precision
    a=1

def fast_text_predict(model_file, txt):
    classifier = ff.load_model(model_file) # 载入已经训练好的模型
    pre = classifier.predict('i like apple', 10)  # 输出改文本的预测结果
    a=1

if __name__ == '__main__':
    train_data_file = './data/fasttext1.train'
    test_data_file = './data/fasttext1.test'
    model_file = './models/fast_model/model'
    fast_text_train(train_data_file, model_file, test_data_file)
    fast_text_predict(model_file, 'C 语言和 C++、C# 的区别在什么地方？')
    a=1