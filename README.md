# A score-based particle method for homogeneous Landau equation

This repository provides an efficient implementation in Pytorch of the score-based particle method for solving the homogeneous Landau equation in collisional plasmas:

$$
\partial_t f(v) = \nabla_v \cdot \int_{\mathbb{R}^d} A(v-v_* ) f(v) f(v_* ) (\nabla_v \log f(v) - \nabla_{v_* } \log f(v_* ) ) dv_*
$$

## Reference
[1] [A score-based particle method for homogeneous Landau equation](https://doi.org/10.1016/j.jcp.2025.114053), Journal of Computational Physics, 2025. 

If you found this repository useful, please consider citing

```
@article{HUANG2025114053,
title = {A score-based particle method for homogeneous Landau equation},
journal = {Journal of Computational Physics},
pages = {114053},
year = {2025},
issn = {0021-9991},
doi = {https://doi.org/10.1016/j.jcp.2025.114053},
url = {https://www.sciencedirect.com/science/article/pii/S0021999125003365},
author = {Yan Huang and Li Wang},
}
```

## Usage
The code "ScoreLandau_BKW.py" is for solving the 2D BKW solution (example 1 in [1]).

The code "ScoreLandau_Coulomb.py" is for solving the 2D anisotropic solution with the Coulomb potential (example 3 in [1]).
