//
//  Neuron.h
//  AI_C++_Test
//
//  Created by maxwell on 6/25/19.
//  Copyright © 2019 organized-organization. All rights reserved.
//

#ifndef Neuron_h
#define Neuron_h
#include "Globals.h"

//Use in a layer class
class Neuron{
private:
    float initVal;
    float activatedVal;
public:
    Neuron(float val);
    
    void activationFunc(void){ this->activatedVal = tanhf(this->initVal); }
    
    float getInitVal(void){ return this->initVal; }
    float getActivatedVal(void){ return this->activatedVal; }
};

typedef struct{
private:
    float initWeight;
    float derivedWeight;
    float weightedOutput;
public:
    //This does all the mathy stuff ya know
    //Also this func will basicly be the whole point of the back prop header file
    void Derive(void);
    void finalOutpFunc(float neuronVal){ this->weightedOutput = derivedWeight * neuronVal; }
    
    float getInitWeight(void){ return this->initWeight; }
    float getDerivativeWeight(void){ return this->derivedWeight; }
    float getWeightedOutp(void){ return this->weightedOutput; }
}Node/*The node is the connection between one neuron to another*/;

class NodeLayers{
private:
    Node *n;
    
public:
    
};

#endif /* Neuron_h */
