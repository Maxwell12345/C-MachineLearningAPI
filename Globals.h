//
//  Globals.h
//  AI_C++_Test
//
//  Created by maxwell on 6/25/19.
//  Copyright © 2019 organized-organization. All rights reserved.
//

#ifndef Globals_h
#define Globals_h
#include <iostream>
#include <cmath>
#include <vector>
#include <array>
#include <ctime>

//Any Global variables u need to make goes here
float setRand(void){
    if ((arc4random() % 10000) % 2 == 0) {
        return (float)(arc4random() % 10000) / 10000.000000;
    }
    else {
        return ((float)(arc4random() % 10000) / 10000.000000) * -1;
    }
}
#define random setRand()

//Note: Plz tell me if u change these, cus they will destroy the whole net if messed with
#define Input_Neuron_Layer_Size 4
#define Hidden1_Neuron_Layer_Size 3
#define Hidden2_Neuron_Layer_Size 3
#define Output_Neuron_Layer_Size 2

#endif /* Globals_h */
