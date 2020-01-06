#include <iostream>
#include <string>
#include <vector>
#include "_eigen/Eigen/Dense"
#include "layer_matrix.hpp"
#include "/Users/idler/Desktop/GitHub/AI_Backbone//Mat_mem/back_propagation.hpp"

template<typename T> static Eigen::Matrix<T, -1, 1>
encode_data2D(T **inp, unsigned *input_shape)
{
    uint32_t i = 0, j = 0, n = 0;
    Eigen::Matrix<T, -1, 1> encoded_data;
    encoded_data.resize((input_shape[1] * input_shape[2]), 1);
    while(i < input_shape[1])
    {
        while(j < input_shape[2])
        {
            encoded_data.row(n).col(0) << inp[i][j];
            j++;
            n++;
        }
        i++;
        j = 0;
    }

    return encoded_data;
}

template<typename T> static Eigen::Matrix<T, -1, 1>
encode_data2D(std::vector<std::vector<T> > inp, unsigned *input_shape)
{
    uint32_t i = 0, j = 0, n = 0;
    Eigen::Matrix<T, -1, 1> encoded_data;
    inp.resize((input_shape[1] * input_shape[2]), 1);
    while(i < input_shape[1])
    {
        while(j < input_shape[2])
        {
            encoded_data.row(n).col(0) << inp[i][j];
            j++;
            n++;
        }
        i++;
        j = 0;
    }

    return encoded_data;
}
static uint32_t iteration = 0;
template<class T>
class Dense
{
public:
    Dense(bool weight_range, unsigned epochs, unsigned *input_shape, bool print);
    inline void initialize_network_input(T **);
    inline void initialize_network_input(std::vector<std::vector<T> >);

    inline void initialize_network_output(T **);
    inline void initialize_network_output(std::vector<std::vector<T> >);

    // inline void initialize_network_input(T **[]);
    inline void initialize_global_variables(void);
    inline void add(unsigned lSize);
    inline void add(unsigned lSize, std::string act_func);
    inline void allocate_network_mem(void);
    inline void train(void);

    void toCons(void);
    
private:
    Layer<T> **network;
    _update_variable_mat_<T> mem;
    unsigned num_layers;
    unsigned *input_shape;
    unsigned epochs;
    
    std::vector<unsigned> lSize_arr;
    std::vector<std::string> act_func_arr;
    bool weight_range;
    bool print;
    std::vector<Eigen::Matrix<T, -1, 1> > input_data;
};

