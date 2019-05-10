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

using json = nlohmann::json;
namespace py = pybind11;
struct CoorPair{
    int x;
    int y;

    CoorPair(){};
    CoorPair(int x0,int y0){x=x0;y=y0;}

};

typedef struct
{
    std::size_t operator() (const CoorPair & CP) const {
            std::size_t h1=std::hash<int>()(CP.x);
            std::size_t h2 = std::hash<int>()(CP.y);
            return h1^(h2<<1);
    }
}CoorPairHash;

typedef struct
{
    bool operator() (const CoorPair & CP1,const CoorPair & CP2) const {
            return (CP1.x==CP2.x)&&(CP1.y==CP2.y);
    }
}CoorPairEqual;


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
    SLIDER
};
struct ParticleState {
    double r[2],F[2];
    double phi;
    double u, v, w;
    int action;
    ParticleType type;
    ParticleState(double x = 0, double y = 0, double phi_0 = 0){
        r[0]=x;r[1]=y;phi=phi_0;
        u = 0;
        v = 0;
        w = 0;
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
    py::array_t<double> get_positions();
    std::vector<double> get_positions_cpp();
    void close();
    json config;
private:
    void read_map();
    void calForces();
    void calForcesHelper_DL(double ri[3], double rj[3], double F[3],int i, int j);    
    bool randomMoveFlag, obstacleFlag, wallFlag, constantPropelFlag;
    double angleRatio;
    static const int dimP = 2;
    static const double kb, T, vis;
    int randomSeed;
    double maxSpeed, maxTurnSpeed;
    std::string configName;
    std::shared_ptr<ParticleState> particle;
    int numObstacles;
    double radius, radius_nm;
    double Bpp; //2.29 is Bpp/a/kT
    double Kappa; // here is kappa*radius
    int numControl;
    double mapRows, mapCols;
    std::string iniFile;
    double dt_, cutoff, mobility, diffusivity_r, diffusivity_t;
    std::default_random_engine rand_generator;
    std::shared_ptr<std::normal_distribution<double>> rand_normal;
    int trajOutputInterval;
    long long timeCounter,fileCounter;
    std::ofstream trajOs;
    std::string filetag;
    bool trajOutputFlag;
    void outputTrajectory(std::ostream& os);
    std::unordered_map<CoorPair, MapSlot, CoorPairHash,CoorPairEqual> mapInfo;
};