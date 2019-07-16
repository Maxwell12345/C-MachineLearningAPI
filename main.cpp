#include <iostream>
#include <vector>
#include <math.h>
#include "Net.hpp"
using namespace std;
Layer *m;
Net::weightStorage *w;

struct a{
    void tester(int a){ a=3; }
};

int main(int argc, const char *argv[]){
    std::cout << argv[0] << std::endl;

    //u uh dum cuz its just the input data shithead
    simulateNetwork(w, m);
    
    return 0;
}