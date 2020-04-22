# https://tensorflow.google.cn/tfx/serving/serving_basic
#TESTDATA="$(pwd)/serving/tensorflow_serving/servables/tensorflow/testdata"
#docker run -t --rm -p 8501:8501 -v "$TESTDATA/saved_model_half_plus_two_cpu:/models/half_plus_two" -e MODEL_NAME=half_plus_two tensorflow/serving &

docker run -t --name senten_class --rm -p 8511:8501 -v "$(pwd)/model:/models/sentence_class" -e MODEL_NAME=sentence_class tensorflow/serving &

#docker run -t --rm -p 8501:8501 -v "/opt/userhome/kdd_zouning/serving/tensorflow_serving/servables/tensorflow/testdata/saved_model_half_plus_two_cpu:/models/half_plus_two" -e MODEL_NAME=half_plus_two tensorflow/serving
