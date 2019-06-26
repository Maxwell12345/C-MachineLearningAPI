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

Neuron::Neuron(float val){
    this->initVal = val;
    activationFunc();
}

void NodeLayers::setInitWeights(){
    
}

void NodeLayers::setDerivedWeights(){
    
}
