#include "../activeParticleSimulator.h"



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

void testDynamicObstacle(){

    ActiveParticleSimulator simulator("config_dynamicObs.json", 1);
    simulator.outputDynamicObstacles();
    simulator.createInitialState(10, 5, 0);
    std::default_random_engine rand_generator;
    std::uniform_real_distribution<double>  rand_unif{0.0, 1.0};
    int step = 200;
    for (int i = 0; i < step; i++){
        std::cout << i << std::endl;
        simulator.updateDynamicObstacles(100);
        double speed = (rand_unif(rand_generator) - 0.5) * 0.1;
        std::vector<double> actions = {speed};
        simulator.run(100, actions);
        //simulator.outputDynamicObstacles();
        //simulator.get_observation_cpp(true);
    
    }

    simulator.outputDynamicObstacles();
    simulator.get_observation_cpp(true);
}



void testDynamicObstacleObservation(){

    ActiveParticleSimulator simulator("config_dynamicObs.json", 1);
    simulator.outputDynamicObstacles();
    simulator.createInitialState(10, 5, 0);
    std::default_random_engine rand_generator;
    std::uniform_real_distribution<double>  rand_unif{0.0, 1.0};
    int step = 5;
    for (int i = 0; i < step; i++){
        std::cout << i << std::endl;
        simulator.updateDynamicObstacles(100);
        simulator.outputDynamicObstacles();
        simulator.get_observation_cpp(true);
    
    }

    
}

void testDynamicObstaclePerformance(){

    ActiveParticleSimulator simulator("config_dynamicObs.json", 1);
    //simulator.outputDynamicObstacles();
    //simulator.createInitialState(10, 5, 0);
    std::default_random_engine rand_generator;
    std::uniform_real_distribution<double>  rand_unif{0.0, 1.0};
    int nRounds = 200;
    for (int j = 0; j < nRounds; j++) {
        std::cout << "cycle: " << j << std::endl;
        simulator.createInitialState(10, 5, 0);
        int step = 200;
        for (int i = 0; i < step; i++){
            simulator.updateDynamicObstacles(100);
            double speed = (rand_unif(rand_generator) - 0.5) * 0.1;
            std::vector<double> actions = {speed};
            simulator.run(100, actions);
            //simulator.outputDynamicObstacles();
            simulator.get_observation_cpp(true);

        }
    }
}

int main(){

    //testSim_Slider();
    //testDynamicObstacle();
    testDynamicObstaclePerformance();
}
