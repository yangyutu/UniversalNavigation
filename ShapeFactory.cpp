/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include "ShapeFactory.h"
#include "activeParticleSimulator.h"


void TrapShapeFactory::initialize() {

    shapeShiftX.clear();
    shapeShiftY.clear();
    

    for (int i = 0; i < 12; i++) {
        for (int j = 0; j < 12; j++) {
            if (shape[i][j] == 1) {
                shapeShiftX.push_back(i - centerX);
                shapeShiftY.push_back(j - centerY);
            }


        }
    }
}

void TrapShapeFactory::fillShape(std::vector<DynamicObstacle>& obstacles) {
    for (int i = 0; i < obstacles.size(); i++) {
        for (int j = 0; j < shapeShiftX.size(); j++) {
            double phi;
            phi = obstacles[i].phi;
            CoorPairDouble cp;
            cp.x = shapeShiftX[j] * cos(phi) + shapeShiftY[j] * sin(phi);
            cp.y = -shapeShiftX[j] * sin(phi) + shapeShiftY[j] * cos(phi);

            obstacles[i].positions.push_back(cp);
        }
    }
}