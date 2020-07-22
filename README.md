# Variant-Selection

Given available hardware resources, our [Augmented Neural Network](https://github.com/Naifeng/Augmented-Neural-Network) (NN+C) can be used to pick the best variant for a given kernel, i.e., picking the best available code among several options. This includes choosing between a CPU and a GPU implementation or identifying compilation flags that will be best suited for the kernel. 

---

About this repository:

`Baseline` contains naive implementations of the blur operation on CPU and GPU.

In `FEniCS`:
* `experiments` contains code for FEniCS CFD solver. Dataset can be generated in two ways: varying the parameter linearly or logistically
* `prediction_for_linear` contains the performance prediction model for the linearly generated dataset
* `prediction_for_log` contains the performance prediction model for the logistically generated dataset
* `results` contains code for plotting the results

In `Halide`:
* `Blur` contains code for Halide Blur function on CPU and GPU
* `FFT` contains code for Halide FFT implementation on CPU and GPU

## Dependencies

for **performance prediction**:
* python >= 3.6.5
* tensorflow >=1.8.0 
* numpy >= 1.16.1
* pandas >= 0.23.0
* scikit-learn >= 0.22.2
* scipy >= 0.16.0

## Building and Running

Details about building and running the code are described in README.md within each folder. Please pay attention to input/output filenames.

## Evaluation

When performance prediction models finish testing, a few metrics will be generated:

* MAE: mean absolute error
* MSE: mean squared error
* MAPE(>*x* s): mean absolute percentage error, restricting to data instances whose runtime is greater than *x* seconds
* rho: Spearman's rank correlation coefficient (or Spearman's rho), which indicates if a higher ranked candidate through prediction also has a higher ranked true runtime. In short, how well does our model tell about the order of runtimes.

*Note: The reason to exclude data instances with small runtime is that sometimes MAPE gets extremely large (>1000%) when the actual runtime is small. In addition, we care more about tasks with a large runtime in the scheduling process.*

## Acknowledgments

* This work is supported by the Defense Advanced Research Projects Agency (DARPA), under the Performant Automation of Parallel Program Assembly (PAPPA) project, at the University of Southern California.

