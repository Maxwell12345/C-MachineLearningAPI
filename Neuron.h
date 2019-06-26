//
//  Neuron.h
//  AI_C++_Test
//
//  Created by maxwell on 6/25/19.
//  Copyright © 2019 organized-organization. All rights reserved.
//

#ifndef Neuron_h
#define Neuron_h

#include <iostream>
#include <vector>
#include <cmath>
#include <stdlib.h>

//
//This just defines the whole neural network
//

//Use in a layer class
class Neuron{
private:
    float initVal;
    float activatedVal;
public:
    Neuron(float val);
    /*Creates exicute system for every Neuron
     
     Neuron::Neuron(float val){
        this->initVal = val;
        activationFunc();
     }Neuron *n1 = new Neuron("Whatever val is needed");
     
     */
    
    void activationFunc(void){ this->activatedVal = tanhf(this->initVal); }
    
    float getInitVal(void){ return this->initVal; }
    float getActivatedVal(void){ return this->activatedVal; }
};

class NeuronLayer{
    
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
    /*
     e.g.
     layerInitWeights[neuron #][weight #]
     */
    std::vector<std::vector<float>> layerInitWeights;
    std::vector<std::vector<float>> layerDerivedWeights;
    
    //Final Output
    std::vector<std::vector<float>> layerWeightedOutp;
    
public:
    std::vector<float> initWeightTempHold;
    std::vector<float> derivativeWeightTempHold;
    std::vector<float> weightedOutpTempHold;
    //In the net header use the Node struct to set and init each of the things
    void setInitWeights(void);
    void setDerivedWeights(void);
    void setweightedOutputs(void);
    
    float getLayerInitWeight(int neuron, int weight){
        return this->layerInitWeights[neuron][weight];
    };
    float getLayerDerivedWeight(int neuron, int weight){
        return this->layerDerivedWeights[neuron][weight];
    };
    float getLayerWeightedOutp(int neuron, int weight){
        return this->layerWeightedOutp[neuron][weight];
    };
    
};

#endif /* Neuron_h */
