LIB_DIRS = 
CC = g++

CDEFS =
CFLAGS = -O0 -g $(INCLUDE_DIRS) $(CDEFS)

CUDA_DIR = /usr/local/cuda
INCLUDE_DIRS = -I/usr/include/opencv4 -I/usr/local/tensorrt/include -I$(CUDA_DIR)/include
LIBS = -L/usr/local/tensorrt/lib -L/usr/lib -lopencv_core -lopencv_flann -lopencv_video -lrt -L$(CUDA_DIR)/lib64 -lcudart -lnvinfer 


HFILES =
CFILES = capture.cpp capture_timed.cpp ipcapture.cpp diffcapture.cpp brighten.cpp gstream_cap.cpp videowriter.cpp gstream_simple.cpp people_detect.cpp

SRCS = ${HFILES} ${CFILES}
OBJS = ${CFILES:.cpp=.o}

all: capture capture_timed ipcapture diffcapture brighten gstream_cap videowriter people_detect

clean:
	-rm -f *.o *.d
	-rm -f capture ipcapture diffcapture brighten gstream_cap capture_timed videowriter people_detect

videowriter: videowriter.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o `pkg-config --libs opencv4` $(LIBS)

gstream_cap: gstream_cap.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o `pkg-config --libs opencv4` $(LIBS)

capture: capture.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o `pkg-config --libs opencv4` $(LIBS)

capture_timed: capture_timed.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o `pkg-config --libs opencv4` $(LIBS)

ipcapture: ipcapture.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o `pkg-config --libs opencv4` $(LIBS)

diffcapture: diffcapture.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o `pkg-config --libs opencv4` $(LIBS)

brighten: brighten.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o `pkg-config --libs opencv4` $(LIBS)

people_detect: people_detect.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o `pkg-config --libs opencv4` $(LIBS)

depend:

.cpp.o: $(SRCS)
	$(CC) $(CFLAGS) -c $<

