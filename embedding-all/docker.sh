# https://tensorflow.google.cn/tfx/serving/serving_basic
#docker run -t --name embedding_entity --rm -p 8511:8501 -v "$(pwd)/models/estimator/1587557924/:/models/embedding_entity" -e MODEL_NAME=embedding_entity tensorflow/serving &

# curl http://192.168.7.218:8512/v1/models/embedding_entity         # 使用curl命令来查看服务的启动状态，也可以看到提供服务的模型版本以及模型状态
# saved_model_cli show --dir ./models/estimator/1587761610/ --all     # TensorFlow提供了一个saved_model_cli命令来查看模型的输入和输出参数（启动python）

docker run -p 8512:8501 \
--mount type=bind,source=/opt/userhome/kdd_zouning/entity_similar/models/estimator/,target=/models/embedding_entity \
-e MODEL_NAME=embedding_entity -t tensorflow/serving &
