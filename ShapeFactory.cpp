/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include "ShapeFactory.h"
#include "activeParticleSimulator.h"


void ShapeFactory::initialize(int shapeWidth){
    
    shapeShiftX.push_back(std::vector<int>());
    shapeShiftY.push_back(std::vector<int>());
            
    for (int i = -shapeWidth; i < shapeWidth; i++){
        for (int j = -shapeWidth; j < shapeWidth; j++) {
            shapeShiftX[0].push_back(i);
            shapeShiftY[0].push_back(j);
        
        }
    }
}

void ShapeFactory::fillShape(std::vector<DynamicObstacle>& obstacles){
    for (int i = 0; i < obstacles.size(); i++){
        int shapeIdx = 0;
        for (int j = 0; j < shapeShiftX[shapeIdx].size(); j++){
            double phi;
            phi = obstacles[i].phi;
            CoorPairDouble cp;
            cp.x = obstacles[i].x + shapeShiftX[shapeIdx][j] * cos(phi) + shapeShiftY[shapeIdx][j] * sin(phi);
            cp.y = obstacles[i].y - shapeShiftX[shapeIdx][j] * sin(phi) + shapeShiftY[shapeIdx][j] * cos(phi);
            
            obstacles[i].positions.push_back(cp);
        }
    }
}
