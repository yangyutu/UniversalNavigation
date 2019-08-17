
/*

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
        .def("setInitialState", &ActiveParticleSimulator::setInitialState);
}
