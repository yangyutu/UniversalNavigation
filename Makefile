CC = gcc
CXX = g++

HOME=/home/yangyutu123/
BOOST_INCLUDE=-I/opt/boost/boost_1_57_0
VPATH = cppTest

DEBUGFLAG=-DDEBUG -g3 -O0 -fPIC
RELEASEFLAG= -O3 -march=native -DARMA_NO_DEBUG
CXXFLAGS=  -std=c++11 $(BOOST_INCLUDE) -D__LINUX   `python-config --cflags` `/home/yangyutu123/anaconda3/bin/python -m pybind11 --includes` 
LDFLAG= -L/opt/OpenBLAS/lib  -llapack -lblas  -pthread -no-pie `python-config --ldflags`

OBJ=testSimulator.o ActiveParticleSimulator.o

test.exe: $(OBJ)
	$(CXX) -o $@ $^ $(LDFLAG) 
	
%.o:%.cpp
	$(CXX) -c $(CXXFLAGS) $(DEBUGFLAG) $^
	




clean:
	rm *.o *.exe
	
