CC = gcc
CXX = g++

HOME=/home/yangyutu/
VPATH = cppTest

DEBUGFLAG=-DDEBUG -g3 -O0 -fPIC
RELEASEFLAG= -O3 -march=native -DARMA_NO_DEBUG
CXXFLAGS=  -std=c++11 $(BOOST_INCLUDE) -D__LINUX  -I/home-4/yyang60@jhu.edu/work/Yang/Downloads/pybind11/include `python-config --cflags` `python -m pybind11 --includes` 
LDFLAG= -L/opt/OpenBLAS/lib  -pthread  `python-config --ldflags`

OBJ=testSimulator.o activeParticleSimulator.o ShapeFactory.o

test.exe: $(OBJ)
	$(CXX) -o $@ $^ $(LDFLAG) 
	
%.o:%.cpp
	$(CXX) -c $(CXXFLAGS) $(DEBUGFLAG) $^
	




clean:
	rm *.o *.exe
	
