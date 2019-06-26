//
//  main.cpp
//  AI_C++_Test
//
//  Created by maxwell on 6/25/19.
//  Copyright © 2019 organized-organization. All rights reserved.
//
#include "Globals.h"
#include "Net.hpp"

//IGNOR EVERYTHING IN HERE IT IS JUST A TESTING BLOCK, LIKE PROOF OF CONCEPT

int main(int argc, const char * argv[]) {
    NodeLayers a;
    
    a.setInitWeights();
    
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 3; j ++) {
            std::cout << a.getLayerInitWeight(i, j) << std::endl;
        }
    }
    for (int i = 0; i < 5; i ++) {
        std::cout << setRand() << std::endl;
    }
    
    return 0;
}
