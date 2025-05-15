# A score-based particle method for homogeneous Landau equation

This repository provides an efficient implementation in Pytorch of the score-based particle method for solving the homogeneous Landau equation in collisional plasmas.

## Reference
[A score-based particle method for homogeneous Landau equation](https://doi.org/10.1016/j.jcp.2025.114053), Journal of Computational Physics, 2025. 

## Usage
This code is for solving the 2D BKW solution of the homogeneous Landau equation:

$$
\partial_t f(v) = \nabla_v \cdot \int_{\mathbb{R}^d} A(v-v_* ) f(v) f(v_* ) (\nabla_v \log f(v) - \nabla_{v_* } \log f(v_* ) ) dv_*
$$
with collision kernel
$$
A(v) = \frac{1}{16} (|v|^2 I_2 - v \otimes v)
$$
