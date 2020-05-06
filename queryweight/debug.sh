#!/bin/bash

LogPath=/opt/userhome/algo/dsxreclog
#LogPath=/opt/userhome/icwork/algo/dsxreclog

#grep -w _company_list_search_ ${LogPath}/* | awk -F'=' '{if($5>1) print $5}'

#grep -w cost ${LogPath}/* | awk -F'=' ' {if($4>"2") print $0"\n"}'

grep -w cost ${LogPath}/* | sed 's/\(.*\) cost=\([0-9.]\+\)s/\1@\2/g' | awk -F "@" '{if($2>2) print$0}'

#grep -w cost ${LogPath}/dsxAPP_recommend.log.2019-08-28 | sed 's/\(.*\) cost=\([0-9.]\+\)s/\1@\2/g' | awk -F "@" '{if($2>1) print$0}'

#grep -w 百度 /opt/userhome/icwork/algo/dsxreclog/dsxAPP_recommend.log.2019-09-30 | awk -F '=' '{if($4>1) print $0}'
