# cuda_schrodinger

Both programs focus on solving the two dimensional Schrödinger equation for a parabolic potential through the finite difference method. 
The CUDA program efficiency might by limitated since these method splits real and imaginary parts of the equation, then the comunication between CPU and GPU is increased.
