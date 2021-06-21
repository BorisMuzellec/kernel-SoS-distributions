# Optimizing Distributions using Kernel Mean Embeddings

## Overview

This repository complements the note [Optimizing Distributions using Kernel Mean Embeddings](https://arxiv.org/abs/2106.09994) (Muzellec B., Rudi, A., Bach, F.):

- `Density fitting.ipynb` is a notebook that implements the density fitting experiments of the paper;
- `optim.py` implements accelerated gradient descent for kernel sos densities;
- `utils.py` contains methods of general utility;
- `data_loaders.py` contains utilities to generate synthetic data.

## References
```
@misc{muzellec2021note,
      title={A Note on Optimizing Distributions using Kernel Mean Embeddings},
      author={Boris Muzellec and Francis Bach and Alessandro Rudi},
      year={2021},
      eprint={2106.09994},
      archivePrefix={arXiv}
}
```

## Dependencies
- Python 3+
- [PyTorch](https://pytorch.org/)
