NVCC := nvcc
NVCC_FLAGS := -O2 -std=c++17
TARGET := cuda_image_processor
SRC := src/main.cu

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)
	rm -rf output/*
