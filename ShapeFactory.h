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
    std::deque<CoorPairDouble> positionHistory;
    static const int capacity = 5;
    DynamicObstacle(double x0, double y0, double phi0, int shapeIdx0){
        x = x0;
        y = y0;
        phi = phi0;
        shapeIdx = shapeIdx0;
        trapFlag = false;
    }
    void store(){
        positionHistory.emplace_back(x, y);
        if (positionHistory.size() == capacity) {
            positionHistory.pop_front();
        }
    }
    std::deque<CoorPairDouble>& history(){return positionHistory;}
    
};



struct TrapShapeFactory{
    
    
    double shape[12][12] = 
    {
        {1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1},
        {1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1},
        {1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1},
        {0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0},
        {0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0},
        {0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0},
        {0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0},
        {0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0},
        {0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0},
        {1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1},
        {1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1},
        {1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1}
    };
    
	const double centerX = 6;
	const double centerY = 6;
	   
    std::vector<double> shapeShiftX;
    std::vector<double> shapeShiftY;
    TrapShapeFactory(){
    }
    void initialize();
    void fillShape(std::vector<DynamicObstacle>& obstacles);
};




#endif /* SHAPEFACTORY_H */

