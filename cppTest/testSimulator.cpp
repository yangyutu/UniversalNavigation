#include "../ActiveParticleSimulator.h"



void testSim_Slider(){
    ActiveParticleSimulator simulator("config.json", 1);

    

    int step = 1000;
    simulator.createInitialState(0.0, 0.0, 0.0);
    

    for(auto i = 0; i < step; ++i){
                
        if(i%2 == 0){
            std::vector<double> actions = {1.0};
            simulator.run(100, actions);
        }else{
            std::vector<double> actions = {-0.5};
            simulator.run(100, actions);
        }
    }
    simulator.close();

}

void testSim_Slider_obs(){
    ActiveParticleSimulator simulator("config_obs.json", 1);

    

    int step = 1000;
    simulator.createInitialState(3.0, 3.0, 0.0);
    

    for(auto i = 0; i < step; ++i){
                
        if(i%2 == 0){
            std::vector<double> actions = {1.0};
            simulator.run(100, actions);
        }else{
            std::vector<double> actions = {-0.5};
            simulator.run(100, actions);
        }
    }
    simulator.close();

}

int main(){

    //testSim_Slider();
    testSim_Slider_obs();
}