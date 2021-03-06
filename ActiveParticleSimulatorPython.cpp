
/*
<%
setup_pybind11(cfg)
cfg['compiler_args'] = ['-std=c++11' ]
cfg['linker_args'] = ['-L/opt/OpenBLAS/lib  -llapack -lblas  -pthread -no-pie']
cfg['include_dirs']= ['-I/home-4/yyang60@jhu.edu/work/Yang/Downloads/json/include']
cfg['sources'] = ['activeParticleSimulator.cpp', 'ShapeFactory.cpp']
%>
*/


#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <string>
#include "activeParticleSimulator.h"
namespace py = pybind11;


PYBIND11_MODULE(ActiveParticleSimulatorPython, m) {    
    py::class_<ActiveParticleSimulator>(m, "ActiveParticleSimulatorPython")
        .def(py::init<std::string, int>())
        .def("createInitialState", &ActiveParticleSimulator::createInitialState)
        .def("step", &ActiveParticleSimulator::step)
    	.def("getPositions", &ActiveParticleSimulator::get_positions)
    	.def("getObservation", &ActiveParticleSimulator::get_observation)
    	.def("getObservationAt", &ActiveParticleSimulator::get_observation_at)
    	.def("checkDynamicTrap", &ActiveParticleSimulator::checkDynamicTrap)
    	.def("checkDynamicTrapAround", &ActiveParticleSimulator::checkDynamicTrapAround)
    	.def("updateDynamicObstacles", &ActiveParticleSimulator::updateDynamicObstacles)
    	.def("storeDynamicObstacles", &ActiveParticleSimulator::storeDynamicObstacles)
        .def("checkSafeHorizontal", &ActiveParticleSimulator::checkSafeHorizontal)
        .def("setInitialState", &ActiveParticleSimulator::setInitialState)
        .def("outputDynamicObstacles",&ActiveParticleSimulator::outputDynamicObstacles);
}
