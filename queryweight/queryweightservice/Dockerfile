# 基于镜像基础
FROM python:3.6.7
#FROM python:3.6.0
#FROM python:3.5.4

# 设置代码文件夹工作目录 /server
WORKDIR /server

# 复制当前代码文件到容器中 /server
ADD . /server

# 安装所需的包

RUN pip install --timeout=120 -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
#RUN pip install tensorflow-1.14.0-cp36-cp36m-manylinux1_x86_64.whl xgboost-1.0.1-py3-none-manylinux1_x86_64.whl scikit_learn-0.22-cp36-cp36m-manylinux1_x86_64.whl
#RUN pip install tensorflow-1.12.0-cp35-cp35m-manylinux1_x86_64.whl #xgboost-1.0.1-py3-none-manylinux1_x86_64.whl #scikit_learn-0.22-cp36-cp36m-manylinux1_x86_64.whl
#RUN pip install --timeout=120 -r requirements.txt
#RUN pip install --timeout=120 xgboost --index https://pypi.mirrors.ustc.edu.cn/simple/


EXPOSE 51658
VOLUME /server/log

RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
RUN echo 'Asia/Shanghai' > /etc/timezone

# Run run_server.sh when the container launches
#ENTRYPOINT ["./run_server.sh"]
CMD ["sh", "run_server.sh"]
