#include "activeParticleSimulator.h"


double const ActiveParticleSimulator::T = 293.0;
double const ActiveParticleSimulator::kb = 1.38e-23;
double const ActiveParticleSimulator::vis = 1e-3;

ActiveParticleSimulator::ActiveParticleSimulator(std::string configName0, int randomSeed0) {
    rand_normal = std::make_shared<std::normal_distribution<double>>(0.0, 1.0);
    randomSeed = randomSeed0;
    configName = configName0;
    std::ifstream ifile(this->configName);
    ifile >> config;
    ifile.close();

    readConfigFile();

    if (obstacleFlag) {
        this->read_map();
    }

}

void ActiveParticleSimulator::readConfigFile() {

    randomMoveFlag = config["randomMoveFlag"];
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
        angleRatio= config["angleRatio"];
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
    

    fileCounter = 0;
    cutoff = config["cutoff"];
    trajOutputFlag = config["trajOutputFlag"];
    obstacleFlag = config["obstacleFlag"];

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
            particle->w = actions[0] * angleRatio*maxSpeed / radius;
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
        randomX = sqrt(2.0 * diffusivity_t * dt_) * (*rand_normal)(rand_generator);
        randomY = sqrt(2.0 * diffusivity_t * dt_) * (*rand_normal)(rand_generator);
        randomPhi = sqrt(2.0 * diffusivity_r * dt_) * (*rand_normal)(rand_generator);

        particle->r[0] += mobility * particle->F[0] * dt_ +
                particle->u * dt_;
        particle->r[1] += mobility * particle->F[1] * dt_ +
                particle->v * dt_;
        particle->phi += particle->w * dt_;



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

void ActiveParticleSimulator::calForces() {
    double r[dimP], dist, F[3];

    for (int k = 0; k < dimP; k++) {
        particle->F[k] = 0.0;
    }


    if (obstacleFlag) {
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
}

void ActiveParticleSimulator::createInitialState(double x, double y, double phi) {

    particle->r[0] = x*radius;
    particle->r[1] = y*radius;
    particle->phi = phi;
    std::stringstream ss;
    std::cout << "model initialize at round " << fileCounter << std::endl;
    ss << this->fileCounter++;
    if (trajOs.is_open() && trajOutputFlag) trajOs.close();
    if (trajOutputFlag)
        this->trajOs.open(filetag + "xyz_" + ss.str() + ".txt");
    this->timeCounter = 0;

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
    os <<  speed << "\t";
    os << speed / (abs(particle->w) + 1e-6) << "\t";
    os << std::endl;


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
            MapSlot slot(i, j);

            if (mapData[i][j] == 1) {
                slot.occupied = true;
                mapInfo[cp] = slot;
            }

        }
        mapOut << std::endl;
    }

    // add boundary obstacles
    for (int i = -1; i <= mapRows; i++) {
        CoorPair cp1(i, -1), cp2(i, mapCols);
        MapSlot slot1(i, -1), slot2(i, mapCols);
        slot1.occupied = true;
        slot2.occupied = true;
        mapInfo[cp1] = slot1;
        mapInfo[cp2] = slot2;
    }

    for (int j = -1; j <= mapCols; j++) {
        CoorPair cp1(-1, j), cp2(mapRows, j);
        MapSlot slot1(-1, j), slot2(mapRows, j);
        slot1.occupied = true;
        slot2.occupied = true;
        mapInfo[cp1] = slot1;
        mapInfo[cp2] = slot2;
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
