
OPENCV_PREFIX = /opt/local
BOOST_INC= /Users/ktuite/Library/boost_1_51_0
BOOST_LIB=/Users/ktuite/Library/boost_1_51_0/stage/lib

INCLUDES = -I$(OPENCV_PREFIX)/include  -I/usr/include -I$(BOOST_INC)


LIBDIRS = -L$(OPENCV_PREFIX)/lib -L$(BOOST_LIB) -L/usr/lib64

LIBS = -lopencv_features2d -lopencv_highgui -lopencv_imgproc -lopencv_objdetect -lopencv_core -lopencv_video -lboost_system -lboost_filesystem


CC=g++
CFLAGS=-c -Wall -O3
SOURCES=  detect_util.cpp svm.cpp detect.cpp 
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=FaceFeatureDetect

all: $(SOURCES) $(EXECUTABLE)
clean: 
	rm -f *.o
	rm FaceFeatureDetect
    
$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(OBJECTS) $(LIBDIRS) -o $@ $(LIBS)

.cpp.o:
	$(CC) $(CFLAGS) $(INCLUDES) $(LIBDIRS) $< -o $@
