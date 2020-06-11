# Variant-Selection

## Overview
Our augmented neural networks can be used to pick the best variant for a given kernel, i.e., picking the best available code among several options. This includes choosing between a CPU and a GPU implementation or identifying compilation flags that will be best suited for the kernel. 

Here, we demonstrated our approach using two examples: Halide Blur function and FEniCS CFD (Computational Fluid Dynamics) solver, both of which have wide applications. 

## About Repo
`Baseline` contains naive implementations of the blur operation on CPU and GPU

In `FEniCS`:
1. `experiments` contains code for FEniCS CFD solver. Dataset can be generated in two ways: varying the parameter linearly and logistically
2. `prediction_for_linear` contains the performance prediction model for the linearly generated dataset
3. `prediction_for_log` contains the performance prediction model for the logistically generated dataset
4. `results` contains code for plotting the results

In `Halide`:
1. `Blur` contains code for Halide Blur function on CPU and GPU
2. `FFT` contains code for Halide FFT implementation on CPU and GPU

Details about running experiments are described in README.md within each folder.

## Dependencies

for performance prediction models

* python >= 3.6.5
* tensorflow >=1.8.0 
* numpy >= 1.16.1
* pandas >= 0.23.0
* scikit-learn >= 0.22.2
* scipy >= 0.16.0



