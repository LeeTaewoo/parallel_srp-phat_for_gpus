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
# Reference:
# [1] Taewoo Lee, Sukmoon Chang, and Dongsuk Yook, "Parallel SRP-PHAT for
#     GPUs," Computer Speech and Language, vol. 35, pp. 1-13, Jan. 2016.
#
CUDA_INSTALL_PATH ?= /usr/local/cuda
CUDA_SAMPLE_PATH ?= /home/twlee/NVIDIA_CUDA-7.0_Samples
CXX := g++
LINK := nvcc
NVCC := nvcc 

# Includes
INCLUDES = -I$(CUDA_SAMPLE_PATH)/common/inc -I$(CUDA_INSTALL_PATH)/include -I.

# Common flags
COMMONFLAGS += $(INCLUDES) -w -DM=8 -DQ=3888 -DTD_GPU=0 -DTD_GPU_CC=13 -DTD_GPU_SRP=41 -DFD_GPU=1 -DFD_GPU_SRP=2
NVCCFLAGS += $(COMMONFLAGS) 
CXXFLAGS += $(COMMONFLAGS) -I/usr/include -lm
CFLAGS += $(COMMONFLAGS)
LIB_CUDA := -L$(CUDA_INSTALL_PATH)/lib64 -lcudart

# CUDA code generation flags
SMS ?= 20 30 35 37 50 52
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif
NVCC_OPTS := $(GENCODE_FLAGS) -use_fast_math -O3 --compiler-options "-O3 --fast-math"
GCC_OPTS := -O3 --fast-math

OBJS = getDeviceInfo.o
TARGET= getDeviceInfo
LINKLINE= $(LINK) $(LIB_CUDA) -o $(TARGET) $(OBJS) 

.SUFFIXES: .cpp .cu .o

$(TARGET): $(OBJS) Makefile
	$(LINKLINE)
getDeviceInfo.o: getDeviceInfo.cu
	$(NVCC) $(NVCCFLAGS) $(NVCC_OPTS) -c $<

clean :
	rm -f ${OBJS} ${TARGET}
