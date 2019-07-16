#ifndef _NET_HPP
#define _NET_HPP
#include <iostream>
#include <vector>
#include <math.h>
#include <ctime>
#include "InputData.hpp"
#include "Net.cpp"

using namespace Net;
void Net::feedForward(Layer *&matrix, uint32_t setNum, bool isRandom){
    matrix = (Layer *)malloc(sizeof(Layer) * 3);
    std::vector<float> input_disposable;

    //initialize the size of each layer
    matrix[0].layerSize = 4;
    matrix[1].layerSize = 3;
    matrix[2].layerSize = 2;

    matrix[0].neuron = (Neuron **)malloc(sizeof(Neuron *) * matrix[0].layerSize);
    matrix[1].neuron = (Neuron **)malloc(sizeof(Neuron *) * matrix[1].layerSize);
    matrix[2].neuron = (Neuron **)malloc(sizeof(Neuron *) * matrix[2].layerSize);

    //Input/Hidden/Output Layers (Output Layer taken care of in weight topology)
    
    for (int n = 0; n < 3; n = n + 1){
        weightTopology(matrix, isRandom, n, setNum);
    }

    printShit(matrix);
}

void simulateNetwork(struct weightStorage *&weight, Layer *&matrix){
    //This is the number of iterations you want the net to run
    for (int i = 0; i <= 17; ++i){
        if (i == 0){feedForward(matrix, i, 1);}
        else {feedForward(matrix, i, 0);}
        setStorage(weight, matrix);
        //updateWeights() -This is what u will do after back_prop

        for (int j = 0; j < 2; ++j){
            for (int n = 0; n < matrix[j].layerSize; n++){
                matrix[j].neuron[n]->~Neuron();
            }
            weight[j].~weightStorage();
        }
    }
}

#endif