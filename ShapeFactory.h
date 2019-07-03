/* 
 * File:   ShapeFactory.h
 * Author: yangyutu
 *
 * Created on June 29, 2019, 2:49 PM
 */

#ifndef SHAPEFACTORY_H
#define SHAPEFACTORY_H

#include<vector>
#include<deque>

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


struct CoorPairDouble{
    double x;
    double y;

    CoorPairDouble(){};
    CoorPairDouble(double x0,double y0){x=x0;y=y0;}

};

struct DynamicObstacle {
    double x, y, phi;
    double vy, trapThresh, speed;
    bool trapFlag;
    int shapeIdx;
    std::vector<CoorPairDouble> positions;
    std::deque<std::vector<CoorPairDouble>> positionHistory;
    static const int capacity = 5;
    DynamicObstacle(double x0, double y0, double phi0, int shapeIdx0){
        x = x0;
        y = y0;
        phi = phi0;
        shapeIdx = shapeIdx0;
    }
    void store(){
        positionHistory.push_back(positions);
        if (positionHistory.size() == capacity) {
            positionHistory.pop_front();
        }
    }
    std::deque<std::vector<CoorPairDouble>>& history(){return positionHistory;}
    
};



struct ShapeFactory{
    
    /*
    int shape[8][8] = 
    {
        {1, 1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1, 1}
    };
    */
        int shape[6][6] = 
    {
        {1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1}
    };
    
    std::vector<std::vector<int>> shapeShiftX;
    std::vector<std::vector<int>> shapeShiftY;
    std::vector<std::vector<int>> centers;
    ShapeFactory(){
    }
    void initialize(int shapeWidth);
    void fillShape(std::vector<DynamicObstacle>& obstacles);
};






#endif /* SHAPEFACTORY_H */

