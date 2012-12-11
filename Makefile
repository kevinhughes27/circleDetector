CC = g++
CFLAGS = -g -Wall
SRCS = circleDetector.cpp
PROG = circleDetector

OPENCV = `pkg-config opencv --cflags --libs`
LIBS += $(OPENCV)

$(PROG):$(SRCS)
	$(CC) $(CFLAGS) -o $(PROG) $(SRCS) $(LIBS) 
