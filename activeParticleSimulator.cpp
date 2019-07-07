#include "activeParticleSimulator.h"


double const ActiveParticleSimulator::T = 293.0;
double const ActiveParticleSimulator::kb = 1.38e-23;
double const ActiveParticleSimulator::vis = 1e-3;

ActiveParticleSimulator::ActiveParticleSimulator(std::string configName0, int randomSeed0) {
    randomSeed = randomSeed0;
    configName = configName0;
    std::ifstream ifile(this->configName);
    ifile >> config;
    ifile.close();

    readConfigFile();
    initializeSensor();
    if (obstacleFlag && !dynamicObstacleFlag) {
        this->read_map();
    } else if (obstacleFlag && dynamicObstacleFlag) {
        for (int n = 0; n < n_channels; n++) {
            mapInfoVec.push_back(std::unordered_map<CoorPair, int, CoorPairHash,CoorPairEqual>());
        }
        
        this->constructDynamicObstacles();
    }
}

void ActiveParticleSimulator::initializeSensor() {

    for (int i = -receptHalfWidth; i < receptHalfWidth + 1; i++) {
        for (int j = -receptHalfWidth; j < receptHalfWidth + 1; j++) {
            sensorXIdx.push_back(i);
            sensorYIdx.push_back(j);
        }
    }
    sensorArraySize = sensorXIdx.size();
    sensorWidth = 2 * receptHalfWidth + 1;
}

bool ActiveParticleSimulator::isValidPosition(double x, double y, double buffer) {

    if ((x < (buffer)) || (x > (wallLength - buffer)) || 
            (y < buffer) || (y > (wallWidth - buffer))) {
        return false;
    }

    return !checkDynamicTrapAround(x, y, buffer);
}

void ActiveParticleSimulator::constructDynamicObstacles() {
    dynamicObstacles.clear();
    double x = 0.0;
    double y;
    double phi = 0.0;
    while (true) { 
        x += dynamicObstacleSpacing;
        y = 0.5 * wallWidth;

        if ((x + 0.5 * dynamicObstacleSpacing) > wallLength) {
            break;
        }
        DynamicObstacle obs(x, y, phi, 0);
        obs.speed = dynamicObsMeanSpeed * (2.0 * rand_uniform(rand_generator) - 1.0); 
        dynamicObstacles.push_back(obs);
        
        phi += M_PI * 0.125;
    }

    shapeFactory.initialize(shapeWidth);
    shapeFactory.fillShape(dynamicObstacles);
    for (int i = 0; i < dynamicObstacles.size(); i++){
        for (int n = 0; n < n_channels; n++){
            dynamicObstacles[i].store();
        }
    }
    
    particle->trapFlag = checkDynamicTrap();

}

void ActiveParticleSimulator::outputDynamicObstacles() {
        // output dynamic obstacles
    std::vector<std::vector<int>> dynamicObsMap(wallLength, std::vector<int>(wallWidth, 0));
    
    for (int i = 0; i < dynamicObstacles.size(); i++) {
        for (int j = 0; j < dynamicObstacles[i].positions.size(); j++) {
            int x_int = dynamicObstacles[i].positions[j].x;
            int y_int = dynamicObstacles[i].positions[j].y;
            if ((x_int >= 0) && (x_int < wallLength) && (y_int >= 0) && (y_int < wallWidth)) {
    
                dynamicObsMap[x_int][y_int] = 1;
            }
        }
    }    

    for (int i = 0; i < wallLength; i++) {
        for (int j = 0; j < wallWidth; j++) {
            std::cout << dynamicObsMap[i][j] << " ";
        }
        std::cout << "\n";
    }
}


void ActiveParticleSimulator::readConfigFile() {

    randomMoveFlag = config["randomMoveFlag"];
    dynamicObstacleFlag = false;

    filetag = config["filetag"];
    //std::vector<double> iniConfig;


    //auto iniConfig = config["iniConfig"];
    //double x = iniConfig[0];

    particle = std::make_shared<ParticleState>(0, 0, 0);
    std::string typeString = config["particleType"];

    if (typeString.compare("FULLCONTROL") == 0) {
        particle->type = ParticleType::FULLCONTROL;
    } else if (typeString.compare("VANILLASP") == 0) {
        particle->type = ParticleType::VANILLASP;
    } else if (typeString.compare("CIRCLER") == 0) {
        particle->type = ParticleType::CIRCLER;
    } else if (typeString.compare("SLIDER") == 0) {
        particle->type = ParticleType::SLIDER;
    } else if (typeString.compare("TWODIM") == 0) {
        particle->type = ParticleType::TWODIM;
    } else {
        std::cout << "particle type out of range" << std::endl;
        exit(2);
    }

    diffusivity_r = 0.161; // characteristic time scale is about 6s
    Tc = 1.0 / diffusivity_r;

    maxSpeed = config["maxSpeed"]; //units of radius per chacteristic time
    radius = config["radius"];
    maxSpeed = maxSpeed * radius / Tc;
    dt_ = config["dt"]; // units of characteristc time
    trajOutputInterval = 1.0 / dt_;
    if (config.contains("trajOutputInterval")) {
        trajOutputInterval = config["trajOutputInterval"];
    }
    circularRadius = 1.0;
    if (config.contains("circularRadius")) {
        circularRadius = config["circularRadius"];
    }
    angleRatio = 1.0;
    if (config.contains("angleRatio") && particle->type == ParticleType::CIRCLER) {
        angleRatio = config["angleRatio"];
        circularRadius = 1.0 / angleRatio;
    }
    circularRadius = circularRadius * radius;
    maxTurnSpeed = maxSpeed / circularRadius;

    dt_ = dt_*Tc;
    diffusivity_t = 2.145e-13; // this corresponds the diffusivity of 1um particle
    diffusivity_t = 2.145e-14; // here I want to manually decrease the random noise
    //diffusivity_r = parameter.diffu_r; // this correponds to rotation diffusity of 1um particle

    Bpp = config["Bpp"];
    Bpp = Bpp * kb * T * 1e9; //2.29 is Bpp/a/kT
    Kappa = config["kappa"]; // here is kappa*radius
    radius_nm = radius * 1e9;
    mobility = diffusivity_t / kb / T;

    // attraction paramters
    //Os_pressure = config["Os_pressure"];
    //Os_pressure = Os_pressure * kb * T * 1e9;
    //L_dep = config["L_dep"]; // 0.2 of radius size, i.e. 200 nm
    //combinedSize = (1 + L_dep) * radius_nm;
    
    fileCounter = 0;
    cutoff = config["cutoff"];
    trajOutputFlag = config["trajOutputFlag"];
    obstacleFlag = config["obstacleFlag"];
  
    staticObstacleTrapThresh = 2.0;
    
    if (config.contains("staticObstacleTrapThresh")) {
        staticObstacleTrapThresh = config["staticObstacleTrapThresh"];
    }
    
    
    if (obstacleFlag && config.contains("dynamicObstacleFlag")) {
        dynamicObstacleFlag = config["dynamicObstacleFlag"];

        wallWidth = config["wallWidth"];
        wallLength = config["wallLength"];
        dynamicObstacleDistThresh = config["dynamicObstacleDistThresh"];
        
        dynamicObstacleSpacing = config["dynamicObstacleSpacing"];
        dynamicObsMeanSpeed = config["dynamicObsMeanSpeed"]; 

        shapeWidth = config["shapeWidth"];
        trapFactor = 1.0;
        if (config.contains("trapFactor"))
            trapFactor = config["trapFactor"];
        
        n_channels = config["n_channels"];
    }
    
    receptHalfWidth = config["receptHalfWidth"];
    this->rand_generator.seed(randomSeed);
}

py::array_t<double> ActiveParticleSimulator::get_positions() {

    std::vector<double> positions(3);

    positions[0] = particle->r[0] / radius;
    positions[1] = particle->r[1] / radius;
    positions[2] = particle->phi;


    py::array_t<double> result(3, positions.data());

    return result;

}

std::vector<double> ActiveParticleSimulator::get_positions_cpp() {
    std::vector<double> positions(3);
    positions[0] = particle->r[0] / radius;
    positions[1] = particle->r[1] / radius;
    positions[2] = particle->phi;
    return positions;


}

void ActiveParticleSimulator::updateDynamicObstacles(int steps) {

    
    
    
    for (int i = 0; i < dynamicObstacles.size(); i++) {
        //trapping obstacles will not move
        if (!dynamicObstacles[i].trapFlag){       
            if (dynamicObstacles[i].y < shapeWidth || dynamicObstacles[i].y >= (wallWidth - shapeWidth)){
                dynamicObstacles[i].speed *= -1.0;
            }
            double move = dynamicObstacles[i].speed * steps * dt_ / Tc;
            dynamicObstacles[i].y += move;

            //for (int j = 0; j < dynamicObstacles[i].positions.size(); j++) {
            //    dynamicObstacles[i].positions[j].y += move;
            //}
        }
        dynamicObstacles[i].store();
    }
    
    
    
    particle->trapFlag = checkDynamicTrap();
    
#ifdef DEBUG
    std::cout << "dynamicTrap: " << particle->trapFlag << std::endl;

#endif
}



bool ActiveParticleSimulator::_checkDynamicTrapAt(double x, double y){
    bool trapFlag = false;
    for (int i = 0; i < dynamicObstacles.size(); i++) dynamicObstacles[i].trapFlag = false;
    
    for (int i = 0; i < dynamicObstacles.size(); i++) {
        double dist = sqrt(pow((dynamicObstacles[i].x - x), 2)
                + pow((dynamicObstacles[i].y - y), 2));

        if (dist < dynamicObstacleDistThresh) {
            for (int j = 0; j < dynamicObstacles[i].positions.size(); j++) {
                double r_obs[2] = {dynamicObstacles[i].positions[j].x + dynamicObstacles[i].x , dynamicObstacles[i].positions[j].y + dynamicObstacles[i].y};
                double dist2 = sqrt(pow((dynamicObstacles[i].positions[j].x - x), 2)
                + pow((dynamicObstacles[i].positions[j].y - y), 2));
                if (dist2 < 2.0) {
                    dynamicObstacles[i].trapFlag = true;
                    return true;
                }

            }
        }
    }
    return trapFlag;
}

bool ActiveParticleSimulator::checkDynamicTrapAround(double x, double y, double buffer){

    for (int i = -1; i < 2; i++){
        for (int j = -1; j < 2; j++) {        
            if (_checkDynamicTrapAt(x + i, y + i)){
                return true;
            }
        }
    }
    return false;
}

bool ActiveParticleSimulator::checkDynamicTrap(){
    return _checkDynamicTrapAt(particle->r[0] / radius, particle->r[1] / radius);
}

void ActiveParticleSimulator::run(int steps, const std::vector<double>& actions) {
    
    for (int stepCount = 0; stepCount < steps; stepCount++) {
        if (((this->timeCounter) == 0) && trajOutputFlag) {
            this->outputTrajectory(this->trajOs);
        }

        calForces();

        if (particle->type == ParticleType::FULLCONTROL) {
            particle->u = actions[0] * maxSpeed * cos(particle->phi);
            particle->v = actions[0] * maxSpeed * sin(particle->phi);
            particle->w = actions[1] * maxTurnSpeed;
        }
        if (particle->type == ParticleType::VANILLASP) {
            particle->u = actions[0] * maxSpeed * cos(particle->phi);
            particle->v = actions[0] * maxSpeed * sin(particle->phi);
            particle->w = 0.0;
        }
        if (particle->type == ParticleType::CIRCLER) {
            particle->u = actions[0] * maxSpeed * cos(particle->phi);
            particle->v = actions[0] * maxSpeed * sin(particle->phi);
            particle->w = actions[0] * angleRatio * maxSpeed / radius;
        }

        if (particle->type == ParticleType::SLIDER) {
            particle->u = maxSpeed * cos(particle->phi);
            particle->v = maxSpeed * sin(particle->phi);
            particle->w = actions[0] * maxTurnSpeed;
        }

        if (particle->type == ParticleType::TWODIM) {
            particle->u = maxSpeed * actions[0];
            particle->v = maxSpeed * actions[1];
            particle->w = 0.0;
        }

        double randomX, randomY, randomPhi;
        double factor = 1.0;
        if (obstacleFlag && dynamicObstacleFlag) {
            if (particle->trapFlag) {
                factor = trapFactor;
            }
        }
        
        
        randomX = sqrt(2.0 * diffusivity_t * dt_) * rand_normal(rand_generator);
        randomY = sqrt(2.0 * diffusivity_t * dt_) * rand_normal(rand_generator);
        randomPhi = sqrt(2.0 * diffusivity_r * dt_) * rand_normal(rand_generator);

        particle->r[0] += (mobility * particle->F[0] * dt_ +
                particle->u * dt_) * factor;
        particle->r[1] += (mobility * particle->F[1] * dt_ +
                particle->v * dt_) * factor;
        particle->phi += (particle->w * dt_) * factor;



        if (randomMoveFlag) {
            particle->r[0] += randomX;
            particle->r[1] += randomY;
            particle->phi += randomPhi;

        }

        if (particle->phi < 0) {
            particle->phi += 2 * M_PI;
        } else if (particle->phi > 2 * M_PI) {
            particle->phi -= 2 * M_PI;
        }

        this->timeCounter++;
        if (((this->timeCounter) % trajOutputInterval == 0) && trajOutputFlag) {
            this->outputTrajectory(this->trajOs);
        }
    }
}


// this force calculation only includes double layer repulsion 

void ActiveParticleSimulator::calForcesHelper_DL(double ri[2], double rj[2], double F[3], int i, int j) {
    double r[dimP], dist;

    dist = 0.0;
    for (int k = 0; k < dimP; k++) {
        F[k] = 0.0;
        r[k] = (rj[k] - ri[k]) / radius;
        dist += pow(r[k], 2.0);
    }
    dist = sqrt(dist);
    if (dist < 2.0) {
        std::cerr << "overlap " << i << "\t with " << j << "\t" << this->timeCounter << "dist: " << dist << std::endl;
        dist = 2.06;
    }
    if (dist < cutoff) {
        double Fpp = -Bpp * Kappa * exp(-Kappa * (dist - 2.0));

        for (int k = 0; k < dimP; k++) {
            F[k] = Fpp * r[k] / dist;
        }
    }
}

// this force calculation includes double layer repulsion and depletion attraction 

void ActiveParticleSimulator::calForcesHelper_DLAO(double ri[2], double rj[2], double F[3], int i, int j) {
    double r[dimP], dist;

    dist = 0.0;
    for (int k = 0; k < dimP; k++) {
        F[k] = 0.0;
        r[k] = (rj[k] - ri[k]) / radius;
        dist += pow(r[k], 2.0);
    }
    dist = sqrt(dist);
    if (dist < 2.0) {
        std::cerr << "overlap " << i << "\t" << j << "\t at" << this->timeCounter << "dist: " << dist << std::endl;
        dist = 2.06;
    }

    if (dist < cutoff) {
        double Fpp = -4.0 / 3.0 *
                Os_pressure * M_PI * (-3.0 / 4.0 * pow(combinedSize, 2.0) + 3.0 * dist * dist / 16.0 * radius_nm * radius_nm);
        Fpp += -Bpp * Kappa * exp(-Kappa * (dist - 2.0));
        for (int k = 0; k < dimP; k++) {
            F[k] = Fpp * r[k] / dist;

        }
    }
}

void ActiveParticleSimulator::calForces() {
    double r[dimP], dist, F[3];

    for (int k = 0; k < dimP; k++) {
        particle->F[k] = 0.0;
    }


    if (obstacleFlag && !dynamicObstacleFlag) {
        int x_int = (int) std::floor(particle->r[0] / radius + 0.5);
        int y_int = (int) std::floor(particle->r[1] / radius + 0.5);

        for (int i = -2; i < 3; i++) {
            for (int j = -2; j < 3; j++) {
                CoorPair cp(x_int + i, y_int + j);

                if (mapInfo.find(cp) != mapInfo.end()) {
                    double r_obs[2] = {(x_int + i) * radius, (y_int + j) * radius};
                    double F[2];
                    calForcesHelper_DL(particle->r, r_obs, F, i, -1);
                    for (int k = 0; k < dimP; k++) {
                        particle->F[k] += F[k];


                    }

                }
            }
        }
    }

    if (obstacleFlag && dynamicObstacleFlag) {
         // wall parameters
        double dist_x = (particle->r[0] / radius - 0.0); // one wall's position is at 0
        particle->F[0] += 2.0 * Bpp * Kappa * exp(-Kappa * (dist_x - 1.0));
        dist_x = (wallLength - particle->r[0] / radius);
        particle->F[0] += -2.0 * Bpp * Kappa * exp(-Kappa * (dist_x - 1.0));
        
        // wall parameters
        double dist_y = (particle->r[1] / radius - 0.0); // one wall's position is at 0
        particle->F[1] += 2.0 * Bpp * Kappa * exp(-Kappa * (dist_y - 1.0));
        dist_y = (wallWidth - particle->r[1] / radius);
        particle->F[1] += -2.0 * Bpp * Kappa * exp(-Kappa * (dist_y - 1.0));

    }
}

void ActiveParticleSimulator::createInitialState(double x, double y, double phi) {

    particle->r[0] = x*radius;
    particle->r[1] = y*radius;
    particle->phi = phi;
    std::stringstream ss;
    std::cout << "model initialize at round " << fileCounter << std::endl;
    ss << this->fileCounter++;
    if (trajOs.is_open() && trajOutputFlag) trajOs.close();
    if (dynamicObsOs.is_open() && trajOutputFlag) dynamicObsOs.close();
    
    if (trajOutputFlag){
        this->trajOs.open(filetag + "xyz_" + ss.str() + ".txt");
    }
    if (trajOutputFlag && obstacleFlag && dynamicObstacleFlag){
        this->dynamicObsOs.open(filetag + "dynamicObs_" + ss.str() + ".txt");
    }
    this->timeCounter = 0;
    
    if (obstacleFlag && dynamicObstacleFlag) {
        constructDynamicObstacles();
    }

}

void ActiveParticleSimulator::close() {
    if (trajOs.is_open()) trajOs.close();
}

void ActiveParticleSimulator::outputTrajectory(std::ostream& os) {


    os << this->timeCounter << "\t";
    for (int j = 0; j < dimP; j++) {
        os << particle->r[j] / radius << "\t";
    }

    os << particle->phi << "\t";
    os << particle->u << "\t";
    os << particle->v << "\t";
    os << particle->w << "\t";
    double speed = sqrt(pow(particle->u, 2) + pow(particle->v, 2)) / radius;
    os << speed << "\t";
    os << speed / (abs(particle->w) + 1e-6) << "\t";
    os << particle->trapFlag << "\t";
    os << std::endl;

    if (obstacleFlag && dynamicObstacleFlag) {
        
        for (int i = 0; i < dynamicObstacles.size(); i++ ) {
        dynamicObsOs << this->timeCounter << "\t";
        dynamicObsOs << i << "\t";
        dynamicObsOs << dynamicObstacles[i].x << "\t";
        dynamicObsOs << dynamicObstacles[i].y << "\t";
        dynamicObsOs << dynamicObstacles[i].phi << "\t";
        dynamicObsOs << dynamicObstacles[i].speed << "\t";
        dynamicObsOs << dynamicObstacles[i].trapFlag << "\t";
        
        dynamicObsOs << std::endl;
        }
        
    
    }
    
}

void ActiveParticleSimulator::read_map() {

    std::string line;
    std::vector<std::vector<int> > mapData;
    std::string mapName = config["mapName"];
    std::ifstream mapfile(mapName + ".txt");
    if (mapfile.is_open()) {
        while (mapfile.good()) {
            while (!std::getline(mapfile, line, '\n').eof()) {

                std::istringstream reader(line);
                std::vector<int> lineData;
                std::string::const_iterator i = line.begin();
                while (!reader.eof()) {
                    double val;
                    reader >> val;
                    if (reader.fail()) break;
                    lineData.push_back(val);
                }
                mapData.push_back(lineData);
            }
        }
    } else {
        std::cout << "Unable to open file." << std::endl;
    }

    mapfile.close();
    mapRows = mapData.size();
    mapCols = mapData[0].size();

    std::ofstream mapOut(mapName + "out.txt");


    for (int i = 0; i < mapRows; i++) {
        for (int j = 0; j < mapCols; j++) {
            mapOut << mapData[i][j] << " ";
            CoorPair cp(i, j);

            if (mapData[i][j] == 1) {

                mapInfo[cp] = 1;
            }

        }
        mapOut << std::endl;
    }

    mapOut.close();
}

void ActiveParticleSimulator::step(int nstep, py::array_t<double>& actions) {

    auto buf = actions.request();
    double *ptr = (double *) buf.ptr;
    int size = buf.size;
    std::vector<double> actions_cpp(ptr, ptr + size);


    run(nstep, actions_cpp);
}

py::array_t<int> ActiveParticleSimulator::get_observation(bool orientFlag) {
    std::vector<int> linearSensorAll(n_channels * sensorArraySize, 0);


    fill_observation(linearSensorAll, orientFlag);

    py::array_t<int> result(n_channels * sensorArraySize, linearSensorAll.data());

    return result;
}

std::vector<int> ActiveParticleSimulator::get_observation_cpp(bool orientFlag) {

    //initialize linear sensor array
    std::vector<int> linearSensorAll(n_channels * sensorArraySize, 0);

    fill_observation(linearSensorAll, orientFlag);

#ifdef DEBUG
    for (int n = 0; n < n_channels; n++) {
        std::cout << "observation at channel " << n << std::endl;
        
        for (int j = 0; j < sensorWidth; j++) {
            for (int k = 0; k < sensorWidth; k++) {
                std::cout << linearSensorAll[n * sensorArraySize + j * sensorWidth + k ] << " ";
            }
            std::cout << "\n";
        }
    }
#endif
        return linearSensorAll;
}

void ActiveParticleSimulator::fill_observation(std::vector<int>& linearSensorAll, bool orientFlag) {

    double phi = 0.0;
    if (orientFlag) {
        phi = particle->phi;
    }
    

    for (int n = 0; n < n_channels; n++){
    
        mapInfoVec[n].clear();
    }
    for (int i = 0; i < dynamicObstacles.size(); i++) {
        double dist = sqrt(pow((dynamicObstacles[i].x - particle->r[0] / radius), 2)
                + pow((dynamicObstacles[i].y - particle->r[1] / radius), 2));


        if (dist < dynamicObstacleDistThresh) {
            auto iter = dynamicObstacles[i].history().crbegin();

            for (int n = 0; n < n_channels; n++){
                for (int j = 0; j < (*iter).size(); j++) {
                int x_int = (int) std::floor((*iter)[j].x + 0.5);
                int y_int = (int) std::floor((*iter)[j].y + 0.5);
                    if (mapInfoVec[n].find(CoorPair(x_int, y_int)) == mapInfoVec[n].end()) {
                        mapInfoVec[n][CoorPair(x_int, y_int)] = 1;
                    }
                }
                iter++;
            }
        }
    }
   

    for (int n = 0; n < n_channels; n++) {
        for (int j = 0; j < sensorArraySize; j++) {
            // transform from local to global
            double x = sensorXIdx[j] * cos(phi) - sensorYIdx[j] * sin(phi) + particle->r[0] / radius;
            double y = sensorXIdx[j] * sin(phi) + sensorYIdx[j] * cos(phi) + particle->r[1] / radius;
            int x_int = (int) std::floor(x + 0.5);
            int y_int = (int) std::floor(y + 0.5);

            int idx = n * sensorArraySize + j;
            if (mapInfoVec[n].find(CoorPair(x_int, y_int)) != mapInfoVec[n].end()) {
                linearSensorAll[idx] = 1;
            }else if (x_int < 0 || x_int > wallLength || y_int < 0 || y_int > wallWidth) {
                linearSensorAll[idx] = 1;
            }

        }
    }
}