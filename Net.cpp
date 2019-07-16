#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include <stdlib.h>
#include "Back_Prop.hpp"
float setRandom(void){
    if((arc4random() % 1000) % 2 == 0){
        return (float)(arc4random() % 1000) / 1000;
    }else{
        return (float)(arc4random() % 1000) / 1000 * -1;
    }
}
#define random setRandom()

inline float add(std::vector<float> inps){
    float returnVal = 0.000000;
    for (int i = 0; i < inps.size(); ++i){
        returnVal += inps[i];
    }
    return returnVal;
}

typedef struct{
    float m_Weight;
    float m_weightedOutp;
    inline void updateWeight(float newWeight){ this->m_Weight = newWeight; }
    inline void setWeightedOutp(float activatedVal){ this->m_weightedOutp = m_Weight * activatedVal; }
}Connection;

class Neuron{
public:
    Neuron(std::vector<float> inps, bool isRandom, uint32_t nextLayerSize);
    explicit Neuron(std::vector<float> inps){ setInps(inps); }
    ~Neuron();
    inline void setActivatedVal(std::vector<float> inps){ this->m_activatedVal = tanhf(add(inps)); }
    float getActivatedVal(void){ return this->m_activatedVal; }

    inline void initConnections(bool isRandom, uint32_t nextLayerSize);
    float getFinalOutp(uint32_t index){ return this->finalOutputs[index].m_weightedOutp; }

    void initWeights(uint32_t nextLayerSize){this->finalOutputs = (Connection *)malloc(sizeof(Connection) * nextLayerSize);}
    float getWeight(uint32_t index){ return this->finalOutputs[index].m_Weight; }
    
private:
    float m_activatedVal;
    Connection *finalOutputs;
    
protected:
    void setInps(std::vector<float> inps){ this->m_activatedVal = add(inps); }
};
Neuron::Neuron(std::vector<float> inps, bool isRandom, uint32_t nextLayerSize){
        setActivatedVal(inps);
        initWeights(nextLayerSize);
        initConnections(isRandom, nextLayerSize);
}
Neuron::~Neuron(){
    free(this->finalOutputs);
    this->m_activatedVal = NULL;
}
inline void Neuron::initConnections(bool isRandom, uint32_t nextLayerSize){
    Connection con_disposable;
    
    if (isRandom){
        srand(5);
        for (int i = 0; i < nextLayerSize; i++){
            con_disposable.m_Weight = (float)random;
            con_disposable.setWeightedOutp(this->getActivatedVal());
            this->finalOutputs[i] = con_disposable;
        }
    }
    else if(!isRandom){
        for (int i = 0; i < nextLayerSize; ++i){
            con_disposable.updateWeight(updateWeight());
            con_disposable.setWeightedOutp(this->getActivatedVal());
            this->finalOutputs[i] = con_disposable;
        }
    }
}

typedef struct{
    //One pointer is for the neuron index and the other is for initializing the constructor
    Neuron** neuron;
    uint32_t layerSize;
}Layer;


////////////////////////// Net ////////////////////////////////////////////

namespace Net{
    Layer *Matrix;
    Neuron *n_disposable;

    struct weightStorage{
        std::vector<float> *layerStorage;
        uint32_t layerSize;
        ~weightStorage(void);
    };
    void setStorage(struct weightStorage *&, Layer *);

    void printShit(Layer *&);
    void weightTopology(Layer *&, bool, uint32_t, uint32_t);
    void feedForward(Layer *&, uint32_t, bool);
};

void Net::weightTopology(Layer *&matrix, bool isRandom, uint32_t index, uint32_t setNum){
    std::vector<float> inputDisposable;

    switch(index){
        case 0:
            for (int i = 0; i < matrix[0].layerSize; ++i){
            inputDisposable.push_back(input_data[setNum][i]);
            //n_disposable = new Neuron(input_disposable, isRansom, matrix[1].layerSize);
            matrix[0].neuron[i] = new Neuron(inputDisposable, isRandom, matrix[1].layerSize);
            //delete n_disposable;
            inputDisposable.pop_back();
        }

        case 1:
            for (int i = 0; i < matrix[1].layerSize; ++i){
                for (int j = 0; j < matrix[0].layerSize; ++j){
                    inputDisposable.push_back(matrix[0].neuron[j]->getFinalOutp(i));
                }
                matrix[1].neuron[i] = new Neuron(inputDisposable, isRandom, matrix[2].layerSize);
                inputDisposable.clear();
            }
        break;

        case 2:
            for (int i = 0; i < matrix[2].layerSize; ++i){
                for (int j = 0; j < matrix[1].layerSize; ++j){
                    inputDisposable.push_back(matrix[0].neuron[j]->getFinalOutp(i));
                }
                matrix[2].neuron[i] = new Neuron(inputDisposable);
                inputDisposable.clear();
            }
        break;

        default:
            std::cout << "ERROR IN CASE VALUE IN TOPOLOGY CLASS" << std::endl;
        break;
    }
}

void Net::printShit(Layer *&matrix){
    for (int n = 0; n < 3; ++n){
        if(n == 0)std::cout << std::endl << "=================\nInput Layer\n=================" << std::endl << std::endl;
        if(n == 1)std::cout << std::endl << "=================\nHidden Layer\n=================" << std::endl << std::endl;
        if(n == 2){
            std::cout << std::endl << "=================\nOutput Layer\n=================" << std::endl << std::endl;
            for (int i = 0; i < matrix[n].layerSize; ++i){
                std::cout << "Activated Val " << i + 1
                << ": " << matrix[n].neuron[i]->getActivatedVal() << std::endl;
            }
            break;
        }

        for (int i = 0; i < matrix[n].layerSize; ++i){
            std::cout << "Activated Val " << i + 1
            << ": " << matrix[n].neuron[i]->getActivatedVal() << std::endl;
        }std::cout << "\n";

    for (int i = 0; i < matrix[n].layerSize; ++i){
        for (unsigned j = 0; j < matrix[n+1].layerSize; ++j){
            std::cout << "Weight    Val " << j+1 << " Of Neuron " << i + 1
            << ": " << matrix[n].neuron[i]->getWeight(j) << std::endl;

            std::cout << "FinalOutp Val " << j+1 << " Of Neuron " << i + 1
            << ": " << matrix[n].neuron[i]->getFinalOutp(j) << std::endl;
            }
            std::cout << "\n";
        }
    }
}

void Net::setStorage(struct weightStorage *&weight, Layer *matrix){
    weight = (weightStorage *)malloc(sizeof(weightStorage) * 2);
    weight[0].layerSize = 12;
    weight[1].layerSize = 6;

    for(int i = 0; i < 2; ++i){
        weight[i].layerStorage=(std::vector<float> *)malloc(sizeof(std::vector<float>) * weight[i].layerSize);
    }

    for (int n = 0; n < 2; ++n){
        for (int i = 0; i < weight[n].layerSize / matrix[n+1].layerSize; ++i){
            for (int j = 0; j < matrix[n].layerSize; ++j){
                weight[n].layerStorage[i].push_back(matrix[n].neuron[i]->getWeight(j));
            }
        }
    }
}

Net::weightStorage::~weightStorage(){
    for (int i = 0; i < this->layerSize; ++i){
        this->layerStorage[i].clear();
    }
    this->layerSize = NULL;
}