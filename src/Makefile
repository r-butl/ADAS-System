# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall -pthread -I/usr/local/include/opencv4 -I$(SERVICES_DIR) -I$(INCLUDE_DIR) `pkg-config --cflags opencv4`
LDFLAGS = `pkg-config --libs opencv4`

# Directories
SRC_DIR = .
SERVICES_DIR = $(SRC_DIR)/services
INCLUDE_DIR = $(SRC_DIR)/include
BUILD_DIR = $(SRC_DIR)/build


# Default target
all: adas_app

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
$(SRC_DIR)/adas_app: $(MAIN_OBJ) $(READ_FRAME_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# Create build directory if it doesn't exist
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Clean rule
clean:
	rm -rf $(BUILD_DIR) 

# Phony targets
.PHONY: all clean