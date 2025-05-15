# A score-based particle method for homogeneous Landau equation

Pytorch implementation of score-based particle method for homogeneous Landau equation.

## Associated Publication
Paper: 

This code is for solving the 2D BKW solution for Maxwell molecules:

$$
\partial_t f = \nabla_v \cdot \int_{\mathbb{R}^d} A(v-v_* ) f f_* (\nabla_v \log f - \nabla_{v_* } \log f_* ) dv_*
$$

with collision kernel

$$
A(v) = \frac{1}{16}  (|v|^2 I_2 - v \otimes v)
$$
