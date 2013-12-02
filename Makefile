
OPENCV_PREFIX = /opt/local
BOOST_INC = /usr/local/Cellar/boost/1.55.0/include
BOOST_LIB = /usr/local/Cellar/boost/1.55.0/lib

INCLUDES = -I$(OPENCV_PREFIX)/include -I$(BOOST_INC)


LIBDIRS = -L$(OPENCV_PREFIX)/lib -L$(BOOST_LIB)

LIBS = -lopencv_features2d -lopencv_highgui -lopencv_imgproc -lopencv_objdetect -lopencv_core -lopencv_video -lboost_system -lboost_filesystem


CC=g++
CFLAGS=-c -Wall -O3
SOURCES=  detect_util.cpp svm.cpp detect.cpp 
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=SVMTrainAndPredict

all: $(SOURCES) $(EXECUTABLE)
clean: 
	rm -f *.o
	rm $(EXECUTABLE)
    
$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(OBJECTS) $(LIBDIRS) -o $@ $(LIBS)

.cpp.o:
	$(CC) $(CFLAGS) $(INCLUDES) $(LIBDIRS) $< -o $@
