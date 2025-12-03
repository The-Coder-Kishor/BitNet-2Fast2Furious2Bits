# BitNet AMD NPU Kernel Build Makefile
#
# This Makefile builds the NPU kernels and generates the xclbin file
# for AMD NPU acceleration.
#
# Prerequisites:
#   - AMD Vitis/Vivado toolchain
#   - MLIR-AIE tools
#   - XRT (Xilinx Runtime)

# Configuration
MLIR_AIE_DIR ?= ../../mlir-aie
AIE_KERNELS_DIR = $(MLIR_AIE_DIR)/aie_kernels/aie2

# Target device (npu or npu2)
DEVICE ?= npu

# Model configuration (bitnet_b1_58-3B by default)
MODEL ?= bitnet_b1_58-3B
LAYER ?= 0

# Tile sizes
M_TILE ?= 32
K_TILE ?= 32
N_CORES ?= 1

# Build directories
BUILD_DIR = build
KERNEL_BUILD_DIR = $(BUILD_DIR)/kernels

# Output files
XCLBIN = $(BUILD_DIR)/bitnet_mv.xclbin
KERNEL_OBJ = $(KERNEL_BUILD_DIR)/mv_ternary.o

# Compiler settings
AIE_CC ?= xchesscc
AIE_CC_FLAGS = -p me -P4 -c

# MLIR tools
MLIR_TRANSLATE ?= mlir-translate
MLIR_OPT ?= mlir-opt
AIE_GENERATE ?= aie-generate-cdo

.PHONY: all clean kernels xclbin mlir test help

all: xclbin

help:
	@echo "BitNet AMD NPU Kernel Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all      - Build everything (default)"
	@echo "  kernels  - Compile AIE kernels"
	@echo "  mlir     - Generate MLIR design"
	@echo "  xclbin   - Generate xclbin package"
	@echo "  test     - Run basic tests"
	@echo "  clean    - Remove build artifacts"
	@echo ""
	@echo "Configuration variables:"
	@echo "  DEVICE   - Target device: npu or npu2 (default: $(DEVICE))"
	@echo "  MODEL    - BitNet model configuration (default: $(MODEL))"
	@echo "  LAYER    - Layer index in model config (default: $(LAYER))"
	@echo "  M_TILE   - M dimension tile size (default: $(M_TILE))"
	@echo "  K_TILE   - K dimension tile size (default: $(K_TILE))"
	@echo "  N_CORES  - Number of AIE cores (default: $(N_CORES))"
	@echo ""
	@echo "Examples:"
	@echo "  make                          # Build with defaults"
	@echo "  make DEVICE=npu2 MODEL=bitnet_b1_58-large"
	@echo "  make clean all"

# Create build directories
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(KERNEL_BUILD_DIR): $(BUILD_DIR)
	mkdir -p $(KERNEL_BUILD_DIR)

# Compile AIE kernels
kernels: $(KERNEL_OBJ)

$(KERNEL_OBJ): kernels/mv_ternary.cc | $(KERNEL_BUILD_DIR)
	@echo "Compiling AIE kernel: mv_ternary.cc"
	$(AIE_CC) $(AIE_CC_FLAGS) \
		-DDIM_M=$(M_TILE) \
		-DDIM_K=$(K_TILE) \
		-I$(AIE_KERNELS_DIR) \
		-o $@ $<

# Generate MLIR design
mlir: $(BUILD_DIR)/bitnet_mv.mlir

$(BUILD_DIR)/bitnet_mv.mlir: bitnet_mv.py | $(BUILD_DIR)
	@echo "Generating MLIR design for $(MODEL) layer $(LAYER)"
	python3 $< \
		--dev $(DEVICE) \
		--model $(MODEL) \
		--layer $(LAYER) \
		--m-tile $(M_TILE) \
		--k-tile $(K_TILE) \
		--n-cores $(N_CORES) \
		> $@

# Generate xclbin (requires full MLIR-AIE toolchain)
xclbin: $(XCLBIN)

$(XCLBIN): $(BUILD_DIR)/bitnet_mv.mlir $(KERNEL_OBJ)
	@echo "Generating xclbin package..."
	@echo "NOTE: Full xclbin generation requires MLIR-AIE toolchain"
	@echo "      This is a placeholder - actual build depends on your setup"
	@if command -v aie2xclbin >/dev/null 2>&1; then \
		aie2xclbin \
			--aie-generate-cdo \
			--aie-generate-npu \
			--no-aie-generate-ctrlpkt \
			--aie-generate-xclbin \
			--xclbin-name=$@ \
			$<; \
	else \
		echo "Warning: aie2xclbin not found, creating placeholder"; \
		touch $@; \
	fi

# Test target
test: $(XCLBIN)
	@echo "Running basic NPU tests..."
	@if [ -f $(XCLBIN) ] && [ -s $(XCLBIN) ]; then \
		echo "xclbin file exists: $(XCLBIN)"; \
		echo "Size: $$(ls -lh $(XCLBIN) | awk '{print $$5}')"; \
	else \
		echo "Warning: xclbin not generated (toolchain may be missing)"; \
	fi

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)

# Model-specific targets
.PHONY: bitnet-3b bitnet-large llama3

bitnet-3b:
	$(MAKE) MODEL=bitnet_b1_58-3B

bitnet-large:
	$(MAKE) MODEL=bitnet_b1_58-large

llama3:
	$(MAKE) MODEL=Llama3-8B-1.58-100B-tokens
