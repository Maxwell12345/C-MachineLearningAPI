//
//  Net.cpp
//  AI_C++_Test
//
//  Created by maxwell on 6/25/19.
//  Copyright © 2019 organized-organization. All rights reserved.
//

#include "Neuron.h"
#include "Back_Propogation.h"

float setRand(void){
    srand(unsigned(time(NULL)));
    if ((arc4random() % 10000) % 2 == 0) {
        return (float)(arc4random() % 10000) / 10000.000000;
    }
    else
        return (float)((arc4random() % 10000) / 10000.000000) * -1;
}
float Random = setRand();

//Note: Plz tell me if u change these, cus they will destroy the whole net if messed with
#define Input_Neuron_Layer_Size 4
#define Hidden_Neuron_Layer_Size_One 3
#define Hidden_Neuron_Layer_Size_Two 3
#define Output_Neuron_Layer_Size 2
#define Number_Of_Inp_Sets 16
//[set num][inp]
static float inputData[16][4]{
    1,0,0,0,
    1,1,0,0,
    1,1,1,0,
    1,1,1,1,
    0,1,0,0,
    0,1,1,0,
    0,1,1,1,
    0,0,1,0,
    0,0,1,1,
    0,0,0,1,
    1,0,1,0,
    0,1,0,1,
    1,0,0,1,
    1,0,1,1,
    1,1,0,1,
    0,0,0,0
};
static float expectedOutps[16]{
    0,
    1,
    0,
    0,
    0,
    1,
    0,
    0,
    1,
    0,
    1,
    1,
    1,
    0,
    0,
    0
};

float add(std::vector<float> inp){
    //To itialize the process
    float return_val = inp[0];
    
    for (int i = 1; i < inp.size(); i += 1) {
        return_val = return_val + inp[i];
    }
    return return_val;
}

Neuron::Neuron(float val){
    this->initVal = val;
    activationFunc();
}

void NeuronLayer::setInitVals(int setNum, int layer){
    switch (layer) {
        case 1:
            //Input Layer
            for (int i = 0; i < Input_Neuron_Layer_Size; i += 1) {
                this->layerInitVals.push_back(inputData[setNum][i]);
            }
            break;
        case 2:
            //Hidden Layer 1
            for (int j = 0; j < Hidden_Neuron_Layer_Size_One; j += 1) {
                NodeLayers hidden1;
                //^ Im gonna change it later to a global static obj pointer(This is a prototype)
                for (int i = 0; i < Input_Neuron_Layer_Size; i += 1) {
                    this->getWeights.push_back(hidden1.getLayerWeightedOutp(i, j));
                } this->layerInitVals.push_back(add(this->getWeights));
            }
            break;
        case 3:
            //Hidden Layer 2
            for (int j = 0; j < Hidden_Neuron_Layer_Size_Two; j += 1) {
                for (int i = 0; i < Hidden_Neuron_Layer_Size_One; i += 1) {
                    NodeLayers hidden2;
                    this->getWeights.push_back(hidden2.getLayerWeightedOutp(i, j));
                } this->layerInitVals.push_back(add(this->getWeights));
            }
            break;
        case 4:
            //Output Layer
            for (int j = 0; j < Output_Neuron_Layer_Size; j += 1) {
                for (int i = 0; i < Hidden_Neuron_Layer_Size_Two; i += 1) {
                    NodeLayers output;
                    this->getWeights.push_back(output.getLayerWeightedOutp(i, j));
                } this->layerInitVals.push_back(add(this->getWeights));
            }
        default:
            std::cout << "Error, in case value \"setInitVals\" function" << std::endl;
            break;
    }
}

void NodeLayers::setInitWeights(){
    
}

void NodeLayers::setDerivedWeights(){
    
}
