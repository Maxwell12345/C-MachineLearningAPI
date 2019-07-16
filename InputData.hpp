#ifndef _INPUTDATA_HPP_
#define _INPUTDATA_HPP_

#include <iostream>

float input_data[17][4] = {
    1, 1, 1, 1,
    1, 1, 1, 0,
    1, 1, 0, 0,
    1, 0, 0, 0,
    0, 1, 1, 1,
    0, 1, 1, 0,
    0, 1, 0, 0,
    0, 0, 0, 0,
    0, 0, 1, 1,
    0, 0, 1, 0,
    0, 0, 0, 1,
    0, 1, 0, 1,
    1, 0, 1, 0,
    1, 0, 0, 1,
    0, 1, 1, 0,
    1, 1, 0, 1,
    1, 0, 1, 1
}; 
float expected_outputs[17] = {
    0,
    0,
    1,
    0,
    0,
    1,
    0,
    0,
    1,
    0,
    0,
    1,
    1,
    1,
    1,
    0,
    0
};
#endif
