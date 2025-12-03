##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# Modified for BitNet NPU acceleration.
# 
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/makefile-common

VPATH := ${srcdir}/../mlir-aie/aie_kernels/aie2:${srcdir}/kernels

device = npu
targetname = bitnet_mv
trace_size = 8192

CHESS ?= true

# BitNet model configuration
MODEL ?= bitnet_b1_58-3B
LAYER ?= 0
M_TILE ?= 32
K_TILE ?= 32
N_CORES ?= 1

aie_py_src = bitnet_mv.py

all: build/final.xclbin build/insts.txt

build/mv_ternary.o: kernels/mv_ternary.cc
	mkdir -p ${@D}
ifeq ($(device),npu)
ifeq ($(CHESS), true)
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -DDIM_M=${M_TILE} -DDIM_K=${K_TILE} -c $< -o ${@F}
else 
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -DDIM_M=${M_TILE} -DDIM_K=${K_TILE} -c $< -o ${@F}
endif
else ifeq ($(device),npu2)
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DDIM_M=${M_TILE} -DDIM_K=${K_TILE} -c $< -o ${@F}
else
	echo "Device type not supported"
endif

build/aie.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< --dev ${device} --model ${MODEL} --layer ${LAYER} \
		--m-tile ${M_TILE} --k-tile ${K_TILE} --n-cores ${N_CORES} > $@

build/aie_trace.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< --dev ${device} --model ${MODEL} --layer ${LAYER} \
		--m-tile ${M_TILE} --k-tile ${K_TILE} --n-cores ${N_CORES} > $@

build/final.xclbin: build/aie.mlir build/mv_ternary.o
	mkdir -p ${@D}
ifeq ($(CHESS), true)
	cd ${@D} && aiecc.py --aie-generate-cdo --no-compile-host --xclbin-name=${@F} \
				--aie-generate-npu --npu-insts-name=insts.txt $(<:%=../%)
else
	cd ${@D} && aiecc.py --aie-generate-cdo --no-compile-host --xclbin-name=${@F} \
		--no-xchesscc --no-xbridge \
				--aie-generate-npu --npu-insts-name=insts.txt $(<:%=../%)
endif

build/final_trace.xclbin: build/aie_trace.mlir build/mv_ternary.o
	mkdir -p ${@D}
ifeq ($(CHESS), true)
	cd ${@D} && aiecc.py --aie-generate-cdo --no-compile-host --xclbin-name=${@F} \
				--aie-generate-npu --npu-insts-name=insts.txt $(<:%=../%)
else
	cd ${@D} && aiecc.py --aie-generate-cdo --no-compile-host --xclbin-name=${@F} \
		--no-xchesscc --no-xbridge \
				--aie-generate-npu --npu-insts-name=insts.txt $(<:%=../%)
endif

build/insts.txt: build/final.xclbin

${targetname}.exe: ${srcdir}/test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake ${srcdir} -DTARGET_NAME=${targetname} -B.
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}.exe $@
else
	cp _build/${targetname} $@ 
endif

run: ${targetname}.exe build/final.xclbin build/insts.txt 
	${powershell} ./$< -x build/final.xclbin -i build/insts.txt -k MLIR_AIE

run_py: build/final.xclbin build/insts.txt
	${powershell} python3 ${srcdir}/test.py -x build/final.xclbin -i build/insts.txt -k MLIR_AIE

trace: ${targetname}.exe build/final_trace.xclbin build/insts.txt 
	${powershell} ./$< -x build/final_trace.xclbin -i build/insts.txt -k MLIR_AIE -t ${trace_size}
	${srcdir}/../mlir-aie/utils/parse_trace.py --filename trace.txt --mlir build/aie_trace.mlir --colshift 1 > trace_bitnet.json

clean_trace:
	rm -rf tmpTrace trace.txt parse*json trace*json

clean: clean_trace
	rm -rf build _build ${targetname}*.exe

.PHONY: all clean clean_trace run run_py trace bitnet-3b bitnet-large llama3 help

bitnet-3b:
	$(MAKE) MODEL=bitnet_b1_58-3B

bitnet-large:
	$(MAKE) MODEL=bitnet_b1_58-large

llama3:
	$(MAKE) MODEL=Llama3-8B-1.58-100B-tokens

help:
	@echo "BitNet AMD NPU Kernel Build System"
	@echo ""
	@echo "Usage: make [target] [options]"
	@echo ""
	@echo "Targets:"
	@echo "  all          - Build xclbin and instructions (default)"
	@echo "  run          - Build and run C++ test"
	@echo "  run_py       - Build and run Python test"
	@echo "  trace        - Build with tracing and run"
	@echo "  clean        - Remove build artifacts"
	@echo "  help         - Show this help"
	@echo ""
	@echo "Model shortcuts:"
	@echo "  bitnet-3b    - Build for bitnet_b1_58-3B"
	@echo "  bitnet-large - Build for bitnet_b1_58-large"
	@echo "  llama3       - Build for Llama3-8B-1.58-100B-tokens"
	@echo ""
	@echo "Options:"
	@echo "  device=npu|npu2   - Target device (default: npu)"
	@echo "  MODEL=<name>      - Model configuration"
	@echo "  LAYER=<n>         - Layer index (default: 0)"
	@echo "  M_TILE=<n>        - M tile size (default: 32)"
	@echo "  K_TILE=<n>        - K tile size (default: 32)"
	@echo "  N_CORES=<n>       - Number of AIE cores (default: 1)"
	@echo "  CHESS=true|false  - Use Chess compiler (default: true)"
