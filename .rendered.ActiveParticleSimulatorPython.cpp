
/*

*/


#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <string>
#include "ActiveParticleSimulator.h"
namespace py = pybind11;


PYBIND11_MODULE(ActiveParticleSimulatorPython, m) {    
    py::class_<ActiveParticleSimulator>(m, "ActiveParticleSimulatorPython")
        .def(py::init<std::string, int>())
        .def("createInitialState", &ActiveParticleSimulator::createInitialState)
        .def("step", &ActiveParticleSimulator::step)
    	.def("getPositions", &ActiveParticleSimulator::get_positions);
}