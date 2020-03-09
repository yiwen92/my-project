echo $1
docker tag ic/queryweight:v$1 hub.ifchange.com/ic/queryweight:v$1
docker push hub.ifchange.com/ic/queryweight:v$1
