//
//  model.hpp
//  AI_Backbone
//
//  Created by maxwell on 12/10/2019.
//  Copyright © 2019 organized-organization. All rights reserved.
//

#ifndef model_hpp
#define model_hpp
#include <iostream>
#include "model.cpp"
#include "_eigen/Eigen/Dense"

template<class T>
Dense<T> :: Dense(bool weight_range, unsigned epochs, unsigned *input_shape, bool print)
{
    this->num_layers = 0;
    this->weight_range = weight_range;
    this->epochs = epochs;
    this->print = print;
    this->network = (Layer<T> **)malloc(sizeof(Layer<T> *));
    this->input_shape = input_shape;
}

template<class T> inline void
Dense<T> :: initialize_global_variables()
{
    //Use this to initialize the back propogation
    uint32_t i = 1;
    this->network[0] = new Layer<T>(this->lSize_arr[0], this->lSize_arr[1], this->act_func_arr[0], this->weight_range);
    this->network[0]->set_NeuronArr1D(this->input_data[0]);
    this->network[0]->init_Mat2D();
    while (i < this->num_layers)
    {
        if(i < this->num_layers - 1)
        {
            this->network[i] = new Layer<T>(this->lSize_arr[i], this->lSize_arr[i + 1], this->act_func_arr[i], this->weight_range);
            this->network[i]->init_Mat2D();
        }
        else
        {
            this->network[i] = new Layer<T>(this->lSize_arr[i], this->act_func_arr[i]);
        }
        
        this->network[i - 1]->feed_forward(this->network[i]);
        i++;
    }

    if(this->print)
    {
        this->toCons();
    }
}

template<class T> inline void
Dense<T> :: initialize_network_input(T **inp)
{
    uint32_t i = 0, n = 0;
    Eigen::Matrix<T, -1, 1> set_disposable;
    set_disposable.resize(this->input_shape[1], 1);
    while(n < this->input_shape[0])
    {
        //this->input_data[n].resize(this->input_shape[1], 1);
        while(i < this->input_shape[1])
        {
            set_disposable.row(i).col(0) << inp[n][i];
            i++;
        }
        this->input_data.push_back(set_disposable);
        set_disposable.setZero();
        n++;
        i = 0;
    }
}

template<class T> inline void
Dense<T> :: initialize_network_input(std::vector<std::vector<T> > inp)
{
    uint32_t i = 0, n = 0;
    //this->input_data = (Eigen::Matrix<T, -1, 1> *)malloc(sizeof(Eigen::Matrix<T, -1, 1>) * this->input_shape[0]);
    Eigen::Matrix<T, -1, 1> set_disposable;
    set_disposable.resize(this->input_shape[1], 1);
    while(n < this->input_shape[0])
    {
        this->input_data[n].resize(this->input_shape[1], 1);
        while(i < this->input_shape[1])
        {
            set_disposable.row(i).col(0) << inp[n][i];
            i++;
        }
        this->input_data.push_back(set_disposable);
        set_disposable.setZero();
        n++;
        i = 0;
    }
}

template<class T> inline void
Dense<T> :: initialize_network_output(T **y_data)
{
    this->mem.format_y_data(y_data);
}

template<class T> inline void
Dense<T> :: initialize_network_output(std::vector<std::vector<T> > y_data)
{
    this->mem.format_y_data(y_data);
}

template<class T> inline void
Dense<T> :: add(unsigned lSize, std::string act_func)
{
    this->num_layers += 1;
    this->network = (Layer<T> **)realloc(this->network, sizeof(Layer<T> *) * this->num_layers);
    this->lSize_arr.push_back(lSize);
    this->act_func_arr.push_back(act_func);
}

template<class T> inline void
Dense<T> :: allocate_network_mem()
{
    uint32_t n = 0;
    this->network = (Layer<T> **)malloc(sizeof(Layer<T> *) * this->num_layers);
    while(n < this->num_layers)
    {
        this->network[n] = (Layer<T> *)malloc(sizeof(Layer<T>));
        n++;
    }
}

template<class T> inline void
Dense<T> :: train()
{
    int data_idx = 1;
    init_mat<T>();
    this->mem.lSize_arr = this->lSize_arr;
    this->mem.record_mat_data(this->network);
    for(int n = 1; n < this->epochs; ++n)
    {
        this->network[0]->set_NeuronArr1D(this->input_data[data_idx]);
        if(data_idx == this->input_shape[0] - 1)
        {
            data_idx = 0;
        }
        else
        {
            data_idx++;
        }
        this->network[0]->set_Mat2D(Layer<T>::format_variable_mat(__W_Mat_Mem__<T>[n - 1][0], this->lSize_arr[0], this->lSize_arr[1]));
        this->network[0]->feed_forward(this->network[1]);

        for(int i = 1; i < this->num_layers - 1; ++i)
        {
            this->network[i]->set_Mat2D(Layer<T>::format_variable_mat(__W_Mat_Mem__<T>[n - 1][i], this->lSize_arr[i], this->lSize_arr[i + 1]));
            this->network[i]->feed_forward(this->network[i + 1]);
        }

        if(this->print)
        {
            this->toCons();
        }
        init_mat<T>();
        this->mem.record_mat_data(this->network);
    }
}

template<class T> void
Dense<T> :: toCons()
{
    iteration++;
    std::cout << "\n\n" << iteration << std::endl;
    for(int i = 0; i < this->num_layers - 1; ++i)
    {
        this->network[i]->toString();
    }
    this->network[this->num_layers - 1]->toString(0);
}

#endif /*model_hpp*/