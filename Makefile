
OPENCV_PREFIX = /usr/pack/opencv-2.3.1-cr/amd64-debian-linux6.0
INCLUDES = -I$(OPENCV_PREFIX)/include \
           -I/usr/include
LIBS = -lopencv_features2d -lopencv_highgui -lopencv_imgproc -lopencv_objdetect -lopencv_core -lopencv_video 
LIBDIRS = -L$(OPENCV_PREFIX)/lib -Wl,-rpath,$(OPENCV_PREFIX)/lib \
          -L/usr/lib \
          -L/lib64
CC=g++
CFLAGS=-c -Wall -O3
SOURCES=  face_feature_detect.cpp svm.cpp detect.cpp 
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=FaceFeatureDetect

all: $(SOURCES) $(EXECUTABLE)
clean: 
	rm -f *.o
	rm FaceFeatureDetect
    
$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(OBJECTS) $(LIBDIRS) $(LIBS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $(INCLUDES) $< -o $@
