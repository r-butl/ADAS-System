# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall -pthread \
	   -I/usr/include/opencv4 \
	   -I$(SERVICES_DIR) \
	   -I$(INCLUDE_DIR) \
	   -I$(DATASTRUCTURES_DIR) \
	   $(shell pkg-config --cflags opencv4) \
	   -I/usr/local/cuda/include
LDFLAGS = $(shell pkg-config --libs opencv4) -L/usr/local/cuda-12/lib64 -lcuda -lnvinfer 

# Directories
SRC_DIR = .
SERVICES_DIR = $(SRC_DIR)/services
DATASTRUCTURES_DIR = $(SRC_DIR)/datastructures
INCLUDE_DIR = $(SRC_DIR)/include
BUILD_DIR = $(SRC_DIR)/build

# Default target
all: adas_app

# Build traffic_lights file
TRAFFIC_LIGHTS_SRC = $(SERVICES_DIR)/traffic_lights.cpp
TRAFFIC_LIGHTS_OBJ = $(BUILD_DIR)/traffic_lights.o
$(TRAFFIC_LIGHTS_OBJ): $(TRAFFIC_LIGHTS_SRC) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Build read_frame file
READ_FRAME_SRC = $(SERVICES_DIR)/read_frame.cpp
READ_FRAME_OBJ = $(BUILD_DIR)/read_frame.o
$(READ_FRAME_OBJ): $(READ_FRAME_SRC) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Build the main file
MAIN_SRC = main.cpp
MAIN_OBJ = $(BUILD_DIR)/main.o
$(MAIN_OBJ): $(MAIN_SRC) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Build the executable
$(SRC_DIR)/adas_app: $(MAIN_OBJ) $(READ_FRAME_OBJ) $(TRAFFIC_LIGHTS_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# Create build directory if it doesn't exist
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Clean rule
clean:
	rm -rf $(BUILD_DIR) 

# Phony targets
.PHONY: all clean
