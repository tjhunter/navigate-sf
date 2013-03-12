CXXFLAGS= -Wall -fPIC -pthread -shared  -fno-omit-frame-pointer -ansi -D_GNU_SOURCE \
           -g -DNDEBUG
PYTHON_INCLUDES=-I/usr/include/python2.7 -I/usr/lib64/python2.7/site-packages/numpy/core/include
LDFLAGS=-Wall -pthread -shared -Wl,--no-undefined -lblas -llapack -lm -lgfortran

all: quic

quic: QUIC.C quic.cpp
	g++ $(CXXFLAGS) -c QUIC.C -o QUIC.o
	g++ $(CXXFLAGS) $(PYTHON_INCLUDES) -I. -c quic.cpp -o quic.o
	g++ -Wall -shared -Wl,-soname,quic.so -Wl,--no-undefined -o quic.so QUIC.o quic.o -lblas -llapack -lm -lgfortran  -lpython2.7 -lboost_python