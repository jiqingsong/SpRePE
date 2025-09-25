
# Reproducing “SpRePE: A Spherical Geometry-Aware Position Embedding for Vision Transformers”

This repository is for reproducing the results of the paper **“SpRePE: A Spherical Geometry-Aware Position Embedding for Vision Transformers”**
The codebase is adapted from: [https://github.com/NVIDIA/makani/tree/main](https://github.com/NVIDIA/makani/tree/main).

## Datasets

Data can be obtained from the following sources.

### Spherical MNIST

* Download MNIST: [https://www.kaggle.com/datasets/hojjatk/mnist-dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)
* Generate Spherical MNIST:

  ```bash
  ./spherical_mnist/generate.sh
  ```

### ERA5

* Download ERA5: [https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5)
