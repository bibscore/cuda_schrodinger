#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>


#define size 2688

double **allocate(){
    int i;
    double **matrix;

    matrix = malloc(size*sizeof(double*));
    for(i = 0; i < size; i++){
        matrix[i] = malloc(size*sizeof(double));
    }

    return(matrix);
}

double complex **allocate_complex(){
    int i;
    double complex **matrix;

    matrix = malloc(size*sizeof(double complex*));
    for(i = 0; i < size; i++){
        matrix[i] = malloc(size*sizeof(double complex));
    }

    return(matrix);
}

void print_matrix(double **matrix){
    int i, j;

    for(i = 0; i < size; i++){
        for(j = 0; j < size; j++){
            printf("%g \t", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

void potential(double **potential){
    int i, j;
    double n = 110;

    for(i = 0; i < size; i++){
        for(j = 0; j < size; j++)
            potential[i][j] = 0.07*i*i + 0.01*j*j;

    }
}

void wave_packet(double complex **p, double **im, double **r, double k0x, double k0y, double x0, double y0, double sigma){
    int x, y, it = 0;

    for(x = 0; x < size; x++){
        for(y = 0; y < size; y++){
            r[x][y] = cos(k0x*x + k0y*y)*exp(-(((x - x0)*(x - x0)) + ((y - y0)*(y - y0)))/(2*(sigma*sigma)));
            im[x][y] = sin(k0x*x + k0y*y)*exp(-(((x - x0)*(x - x0)) + ((y - y0)*(y - y0)))/(2*(sigma*sigma)));
            p[x][y] = r[x][y] + I*im[x][y];
        }

    }
}

void finite_difference(double complex**p, double **im, double **r, double **potential, double alpha, double dt){
    int i, j;
    
    for(i = 1; i < size - 1; i++){
        for(j = 1; j < size - 1; j++){
            r[i][j] = r[i][j] + 2*((4*alpha + 1/2*dt*potential[i][j])*im[i][j] - alpha*(im[i + 1][j] + im[i - 1][j] + im[i][j + 1] + im[i][j - 1]));
            im[i][j] = im[i][j] - 2*((4*alpha + 1/2*dt*potential[i][j])*r[i][j] + alpha*(r[i + 1][j] + r[i - 1][j] + r[i][j + 1] + r[i][j - 1]));
            p[i][j] = r[i][j] + I*im[i][j];
        }
    }
}

int main(){
    double complex **psi;
    double **im_psi, **r_psi, **v;
    double pi = 2.0*atan(1.0), dx = 0.1, dy = 0.1, x0 = 10.0, y0 = 10.0, sigma = 2.0;
    double k0x = 6.6*pi, k0y = 6.6*pi, dt=dx*dx/10.0, alpha = dt/2.0*dx*dx;
    int x, y;
    int dti, tf=10000;
    FILE *data;

    psi = allocate_complex();
    im_psi = allocate();
    r_psi = allocate();
    v = allocate();

    potential(v);

    wave_packet(psi, im_psi, r_psi, k0x, k0y, x0, y0, sigma);

    data = fopen("serial.txt", "w");

    for(dti = 0; dti < tf; dti++){
        finite_difference(psi, im_psi, r_psi, v, alpha, dti);
    }

    fprintf(data, "x    y   psi\n");

    for(x = 0; x < size; x++){
        for(y = 0; y < size; y++){
            fprintf(data, "%d   %d  %0.6g\n", x, y, fabs(((float) psi[x][y]))*fabs(((float) psi[x][y])*dx*dy));
        }
        fprintf(data, "\n");
    }
    fprintf(data, "\n");


    fclose(data);

    free(psi);
    free(im_psi);
    free(r_psi);
    free(v);

    return 0;
}
