#!/bin/bash
# Release date: June 2015
# Author: Taewoo Lee, (twlee@speech.korea.ac.kr)
#
# Copyright (C) 2015 Taewoo Lee
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
# Out: ./exp_total/result/log/[expM,expQ]/*.log
#
CheckError () {
  if [ $? -ne 0 ]; then
    exit 0
	fi
}

cd ./exp_total/getDeviceInfo
CheckError
make clean && make
./getDeviceInfo > ./deviceInfo.txt
CheckError
cd ../tables
CheckError
./copyTables.sh
CheckError
cd ../datFileLists
CheckError
./datFileListGen.sh
CheckError
cd ../makeFiles
CheckError
./makefileGen.sh
CheckError
cd ../exes
CheckError
./exeGen.sh
CheckError
cd ../result
CheckError
./run.sh
CheckError

