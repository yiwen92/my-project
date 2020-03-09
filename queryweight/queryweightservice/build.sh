docker rm -f query_weight
docker kill query_weight
#docker build -t ic/queryweight:v$1 --no-cache .
docker build -t ic/queryweight:v$1 .
docker run -d --name query_weight -v /opt/userhome/algo/queryweightlog:/server/log -p 51658:51658 --net='host' ic/queryweight:v$1
#docker run -it --name query_weight -v /opt/userhome/algo/queryweightlog:/server/log -p 51658:51658 --net='host' ic/queryweight:v$1 bash


#docker run -it --name query_weight -v /opt/userhome/algo/queryweightlog:/server/log -p 51658:51658 --net='host' ic/queryweight:v$1 bash
#docker exec -it query_weight
