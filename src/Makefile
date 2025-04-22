# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall -pthread `pkg-config --cflags opencv4` -I$(SERVICES_DIR) -I$(INCLUDE_DIR)
LDFLAGS = `pkg-config --libs opencv4`

# Directories
SRC_DIR = .
SERVICES_DIR = $(SRC_DIR)/services
INCLUDE_DIR = $(SRC_DIR)/include
BUILD_DIR = $(SRC_DIR)/build

READ_FRAME_SRC = $(SERVICES_DIR)/read_frame.cpp
READ_FRAME_OBJ = $(BUILD_DIR)/read_frame.o

# Default target
all: $(READ_FRAME_OBJ)

# Create build directory if it doesn't exist
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)


$(READ_FRAME_OBJ): $(READ_FRAME_SRC) | $(BUILD_DIR)
	@echo "Building $@ from $<"
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean rule
clean:
	rm -rf $(BUILD_DIR) 

# Phony targets
.PHONY: all clean