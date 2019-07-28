#pragma once
#include<vector>
#include<memory>
#include<random>
#include <string>
#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>
#include <unordered_map>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "ShapeFactory.h"


using json = nlohmann::json;
namespace py = pybind11;



struct MapSlot{
    int x, y;
    bool occupied;
    MapSlot():x(0),y(0){}
    MapSlot(int x0, int y0){
        x = x0;
        y = y0;
        occupied = false;
    }
};



enum class ParticleType {
    FULLCONTROL,
    VANILLASP,
    CIRCLER,
    SLIDER,
    TWODIM
};
struct ParticleState {
    double r[2],F[2];
    double phi;
    double u, v, w;
    int action;
    bool trapFlag;
    ParticleType type;
    ParticleState(double x = 0, double y = 0, double phi_0 = 0){
        r[0]=x;r[1]=y;phi=phi_0;
        u = 0;
        v = 0;
        w = 0;
        trapFlag = false;
    }
};
class ActiveParticleSimulator{
public:

   
    ActiveParticleSimulator(std::string configName, int randomSeed = 0);
    ~ActiveParticleSimulator() 
    {
        trajOs.close();
    }
    void runHelper();
    void run(int steps, const std::vector<double>& actions);
    void createInitialState(double x, double y, double phi);
    void readConfigFile();
    void step(int nstep, py::array_t<double>& actions);
    bool get_particleDyanmicTrapFlag(){ return particle->trapFlag;}
    bool checkDynamicTrap();
    bool _checkDynamicTrapAt(double x, double y);
    bool checkDynamicTrapAround(double x, double y, double bufferX, double bufferY);
    bool checkSafeHorizontal(double x, double bufferX);
    void setInitialState(double x, double y, double phi);

    bool isValidPosition(double x, double y, double buffer);
    void initializeSensor();
    py::array_t<double> get_positions();
    std::vector<double> get_positions_cpp();
    std::vector<int> get_observation_cpp(bool orientFlag);
    py::array_t<int> get_observation(bool orientFlag);
    
    void outputDynamicObstacles();
    void updateDynamicObstacles(int steps);
    void storeDynamicObstacles();
    void close();
    json config;
private:
    void read_map();
    void calForces();
    void calForcesHelper_DL(double ri[2], double rj[2], double F[2],int i, int j);    
    void calForcesHelper_DLAO(double ri[2], double rj[2], double F[2],int i, int j);    
    void constructDynamicObstacles();
    void fill_observation(std::vector<int>& linearSensorAll, bool orientFlag);
    std::vector<DynamicObstacle> dynamicObstacles;
    TrapShapeFactory shapeFactory;
    bool randomMoveFlag, obstacleFlag, wallFlag, constantPropelFlag, dynamicObstacleFlag;
    double angleRatio, circularRadius;
    double dynamicObstacleDistThresh, staticObstacleTrapThresh, wallWidth, wallLength, dynamicObstacleSpacing, dynamicObsMeanSpeed;
    double Os_pressure, L_dep, combinedSize;
    static const int dimP = 2;
    static const double kb, T, vis;
    int randomSeed, n_channels, receptHalfWidth, sensorArraySize, sensorWidth;
    std::vector<int> sensorXIdx, sensorYIdx;
    double maxSpeed, maxTurnSpeed, trapFactor;
    std::string configName;
    std::shared_ptr<ParticleState> particle;
    int numObstacles;
    double radius, radius_nm;
    double Bpp; //2.29 is Bpp/a/kT
    double Kappa; // here is kappa*radius
    int numControl;
    double mapRows, mapCols;
    std::string iniFile;
    double dt_, cutoff, mobility, diffusivity_r, diffusivity_t, Tc;
    std::default_random_engine rand_generator;
    std::normal_distribution<double> rand_normal{0.0, 1.0};
    std::uniform_real_distribution<double> rand_uniform{0.0, 1.0};
    
    int trajOutputInterval, shapeWidth;
    long long timeCounter,fileCounter;
    std::ofstream trajOs, dynamicObsOs;
    std::string filetag;
    bool trajOutputFlag;
    void outputTrajectory(std::ostream& os);
    std::unordered_map<CoorPair, int, CoorPairHash,CoorPairEqual> mapInfo;
    std::vector<std::unordered_map<CoorPair, int, CoorPairHash,CoorPairEqual>> mapInfoVec;
    
};
