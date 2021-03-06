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
# Out: exp_M?_Q?_TD?_CC?_SRP?_FD?_SRP?
#
CheckError () {
  if [ $? -ne 0 ]; then
    exit 0
	fi
}

cp ./codes/* ../../
rm -f ./exp_*
CheckError

# exp_M
for q in 97200; do
  for m in 8 16 32; do
    
	  # FD_GPU
	  for fdgpu in 2 4; do
      echo "FDGPU""$fdgpu"" Q=""$q"" M=""$m"
		  rm -f ../../Makefile
		  CheckError
		  # copy makefile 
		  cp ../makeFiles/"Makefile_M""$m""_Q""$q""_TD0_CC0_SRP0_FD1_SRP""$fdgpu"".txt" ../../Makefile
		  CheckError
		  # compile
		  cd ../../
		  CheckError
		  make clean && make 
		  CheckError
		  # move result file to current folder
		  mv ./exp ./exp_total/exes/"exp_M""$m""_Q""$q""_TD0_CC0_SRP0_FD1_SRP""$fdgpu"
		  CheckError
		  cd ./exp_total/exes
		  CheckError
	  done

	  #TD_GPU
	  for tdcc in 3; do       #3=Proposed
      for tdsrp in 43; do   #43=Proposed
        echo "TDGPU CC=""$tdcc"" SRP=""$tdsrp"" Q=""$q"" M=""$m"
	      rm -f ../../Makefile
	      CheckError
	      cp ../makeFiles/"Makefile_M""$m""_Q""$q""_TD1_CC""$tdcc""_SRP""$tdsrp""_FD0_SRP0"".txt" ../../Makefile
	      CheckError
	      cd ../../
	      CheckError
	      make clean && make
	      CheckError
	      mv ./exp ./exp_total/exes/"exp_M""$m""_Q""$q""_TD1_CC""$tdcc""_SRP""$tdsrp""_FD0_SRP0"
	      CheckError
	      cd ./exp_total/exes
	      CheckError
	    done
	  done
	  
	  for tdcc in 13; do      #13=Minotto
      for tdsrp in 41; do   #41=Minotto
        echo "TDGPU CC=""$tdcc"" SRP=""$tdsrp"" Q=""$q"" M=""$m"
	      rm -f ../../Makefile
	      CheckError
	      cp ../makeFiles/"Makefile_M""$m""_Q""$q""_TD1_CC""$tdcc""_SRP""$tdsrp""_FD0_SRP0"".txt" ../../Makefile
	      CheckError
	      cd ../../
	      CheckError
	      make clean && make
	      CheckError
	      mv ./exp ./exp_total/exes/"exp_M""$m""_Q""$q""_TD1_CC""$tdcc""_SRP""$tdsrp""_FD0_SRP0"
	      CheckError
	      cd ./exp_total/exes
	      CheckError
	    done
	  done
	  
  done
done

# exp_Q
for q in 3888 97200 388800; do
  echo $q

	# FD_GPU
	for fdgpu in 2 4; do
    echo "FDGPU""$fdgpu"" Q=""$q"" M=16"
		rm -f ../../Makefile
		CheckError
		cp ../makeFiles/"Makefile_M16_Q""$q""_TD0_CC0_SRP0_FD1_SRP""$fdgpu"".txt" ../../Makefile
		CheckError
		cd ../../
		CheckError
		make clean && make
		CheckError
		mv ./exp ./exp_total/exes/"exp_M16_Q""$q""_TD0_CC0_SRP0_FD1_SRP""$fdgpu"
		CheckError
		cd ./exp_total/exes			
		CheckError
	done
	
	#TD_GPU
	for tdcc in 13; do      # Minotto (CC=13)
    echo "TDGPU CC=""$tdcc"" SRP=41"" Q=""$q"" M=16"
		rm -f ../../Makefile
		CheckError
		cp ../makeFiles/"Makefile_M16_Q""$q""_TD1_CC""$tdcc""_SRP41_FD0_SRP0"".txt" ../../Makefile
		CheckError
		cd ../../
		CheckError
		make clean && make
		CheckError
		mv ./exp ./exp_total/exes/"exp_M16_Q""$q""_TD1_CC""$tdcc""_SRP41_FD0_SRP0"
		CheckError
		cd ./exp_total/exes			
		CheckError
	done

	for tdsrp in 43; do   # Proposed (SRP=43)
    echo "TDGPU CC=3 SRP=""$tdsrp"" Q=""$q"" M=16"
		rm -f ../../Makefile
		CheckError
		cp ../makeFiles/"Makefile_M16_Q""$q""_TD1_CC3_SRP""$tdsrp""_FD0_SRP0"".txt" ../../Makefile
		CheckError
		cd ../../
		CheckError
		make clean && make
		CheckError
		mv ./exp ./exp_total/exes/"exp_M16_Q""$q""_TD1_CC3_SRP""$tdsrp""_FD0_SRP0"
		CheckError
		cd ./exp_total/exes			
		CheckError
	done
done

rm -f ../../main.cu
rm -f ../../main.h
rm -f ../../main.o
rm -f ../../Makefile
rm -f ../../*.bin
rm -f ../../*.txt

