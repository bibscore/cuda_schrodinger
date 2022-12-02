#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include "cuda_runtime.h"
#include <time.h>

#define size 8640
#define time 400

#define dx 0.1
#define dy 0.1
#define dt 0.001

#define x0 10.0
#define y0 10.0
#define alpha 0.05
#define sigma 2.0

#define pi 2.0*atan(1.0)
#define k0x 6.6*pi
#define k0y 6.6*pi

#define HANDLE_ERROR(call)					                                            \
{                                                                                       \
    cudaError_t error = call;                                                           \
    if(error != cudaSuccess){                                                           \
        fprintf(stderr, "error: %s:%d, ", __FILE__, __LINE__);                          \
        fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error));    \
    }                                                                                   \
}

__global__ void finite_difference_kernel(double *real, double *real_past, double *real_future, double *imag, double *imag_past, double *imag_future, double *potential)
{
    int i, j;

    i = blockIdx.x;
    j = threadIdx.x;

    if((i > 0 && i < size - 1) && (j > 0 && j < size - 1)){
        real_future[i * blockDim.x + j] = real_past[i * blockDim.x + j] + 2*((4*alpha + 1/2*dt*potential[i * blockDim.x + j])*imag[i * blockDim.x + j] - alpha*imag[(i + 1) * blockDim.x + j] + imag[(i - 1) * blockDim.x + j] + imag[i * blockDim.x + (j + 1)] + imag[i * blockDim.x + (j - 1)]);
        imag_future[i * blockDim.x + j] = imag_past[i * blockDim.x + j] + 2*((4*alpha + 1/2*dt*potential[i * blockDim.x + j])*real[i * blockDim.x + j] - alpha*real[(i + 1) * blockDim.x + j] + real[(i - 1) * blockDim.x + j] + real[i * blockDim.x + (j + 1)] + real[i * blockDim.x + (j - 1)]);
    }
}

double *host_allocate()
{
    double *matrix;

    matrix = (double*)calloc((size*size), sizeof(double));

    return(matrix);
}

double complex *host_allocate_complex()
{
    double complex *matrix;

    matrix = (double complex*)calloc((size*size), sizeof(double complex));

    return(matrix);
}

void archiving(double complex *matrix)
{
	int i, j;
	double psisquared;
	FILE *data;

	data = fopen("cudaresults.txt", "w");

	fprintf(data, "x	y	psi\n");
	for(i = 0; i < size; i++){
		for(j = 0; j < size; j++){
			psisquared = cabsf(matrix[i * size + j]*matrix[i * size + j]);
			fprintf(data, "%d	%d	%0.6g\n", i, j, psisquared*dx*dy);
		}
		fprintf(data, "\n");
	}
	fprintf(data, "\n");

	fclose(data);
}

void potential(double *potential)
{
    int i, j;

    for(i = 0; i < size; i++){
        for(j = 0; j < size; j++)
            potential[i * size + j] = 0.07*i*i + 0.01*j*j;
    }
}

void wave_packet(double complex *psi, double *img, double *real)
{
	int i, j;

	for(i = 0; i < size; i++)
	{
		for(j = 0; j < size; j++)
		{
			real[i * size + j] = cos(k0x*i + k0y*j)*exp(-(((i - x0)*(i - x0)) + ((j - y0)*(j - y0)))/(2*(sigma*sigma)));
 	                img[i * size + j] = sin(k0x*i + k0y*j)*exp(-(((i - x0)*(i - x0)) + ((j - y0)*(j - y0)))/(2*(sigma*sigma)));
        	        psi[i * size + j] = real[i * size + j] + I*img[i * size + j];
		}
	}
}

void time_conditioning(double *matrix, double *matrix_past){
    int i, j;

    for(i = 0; i < size; i++){
        for(j = 0; j < size; j++){
            matrix_past[i * size + j] = matrix[i * size + j];
        }
    }
}

void time_dependence(double *past, double *present, double *future){
	int i, j;

	for(i = 0; i < size; i++){
		for(j = 0; j < size; j++){
			past[i * size + j] = present[i * size + j];
			present[i * size + j] = future[i * size + j];			
		}
	}
}

void device_info(){

    cudaDeviceProp prop;
    int i, count;
    FILE *info;

    cudaGetDeviceCount(&count);

    info = fopen("device_info_lncc.txt", "w");

    for(i = 0; i < count; i++){
        cudaGetDeviceProperties(&prop, i);
        fprintf(info, "         device %d information \n", i);
        fprintf(info, "name: %s\n", prop.name);
        fprintf(info, "compute capability: %d.%d\n", prop.major, prop.minor);
        fprintf(info, "clock rate: %d\n", prop.clockRate);
        fprintf(info, "device copy overlap: ");

        if(prop.deviceOverlap)
            fprintf(info, "enabled\n");
        else
            fprintf(info, "disabled\n");

        fprintf(info, "\nkernel execution timeout:\n");
        if(prop.kernelExecTimeoutEnabled)
            fprintf(info, "enableb\n");
        else
            fprintf(info, "disabled\n");

        fprintf(info, "\n       device memory information\n");
        fprintf(info, "total global memory: %ld\n", prop.totalGlobalMem);
        fprintf(info, "total constant memory: %ld\n", prop.totalConstMem);
        fprintf(info, "max memory pitch: %ld\n", prop.memPitch);
        fprintf(info, "texture alignment: %ld\n", prop.textureAlignment);

        fprintf(info, "\n       device multiprocessor information\n");
        fprintf(info, "multiprocessor count: %d\n", prop.multiProcessorCount);
        fprintf(info, "shared memory per multiprocessor: %ld\n", prop.sharedMemPerBlock);
        fprintf(info, "register per multiprocessor: %ld\n", prop.regsPerBlock);
        fprintf(info, "threads in warp: %d\n", prop.warpSize);
        fprintf(info, "max thread dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0],
        prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        fprintf(info, "max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0],
        prop.maxGridSize[1], prop.maxGridSize[2]);

        fprintf(info, "\n\n");

    }
}

int main()
{
    int dti, i, j;
    double complex *host_psi;
    double *host_img, *host_real, *host_potential, *device_img, *device_real, *device_potential;
    double *host_img_past, *host_img_future, *host_real_past, *host_real_future, *device_img_past, *device_img_future, *device_real_past, *device_real_future;
	FILE *final_time;

    device_info();

    clock_t beginTime = clock();

    host_psi = host_allocate_complex();
    host_img = host_allocate();
    host_img_past = host_allocate();
    host_img_future = host_allocate();
    host_real = host_allocate();
    host_real_past = host_allocate();
    host_real_future = host_allocate();
    host_potential = host_allocate();

    HANDLE_ERROR(cudaMalloc((void**) &device_img, (size*size)*sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**) &device_img_past, (size*size)*sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**) &device_img_future, (size*size)*sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**) &device_real, (size*size)*sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**) &device_real_past, (size*size)*sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**) &device_real_future, (size*size)*sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**) &device_potential, (size*size)*sizeof(double)));

    potential(host_potential);
    wave_packet(host_psi, host_img, host_real);
    time_conditioning(host_real, host_real_past);
    time_conditioning(host_real, host_real_future);
    time_conditioning(host_img, host_img_past);
    time_conditioning(host_img, host_img_future);

    for(dti = 0; dti < time; dti++){
        // copying information from host to device
        HANDLE_ERROR(cudaMemcpy(device_real, host_real, (size*size)*sizeof(double), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(device_real_past, host_real_past, (size*size)*sizeof(double), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(device_real_future, host_real_future, (size*size)*sizeof(double), cudaMemcpyHostToDevice));

        HANDLE_ERROR(cudaMemcpy(device_img, host_img, (size*size)*sizeof(double), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(device_img_past, host_img_past, (size*size)*sizeof(double), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(device_img_future, host_img_future, (size*size)*sizeof(double), cudaMemcpyHostToDevice));

        HANDLE_ERROR(cudaMemcpy(device_potential, host_potential, (size*size)*sizeof(double), cudaMemcpyHostToDevice));

        // calling method
        finite_difference_kernel<<<size, size>>>(device_real, device_real_past, device_real_future, device_img, device_img_past, device_img_future, device_potential);


        // copying information from device to host
        HANDLE_ERROR(cudaMemcpy(device_real, host_real, (size*size)*sizeof(double), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(device_real_past, host_real_past, (size*size)*sizeof(double), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(device_real_future, host_real_future, (size*size)*sizeof(double), cudaMemcpyDeviceToHost));

        HANDLE_ERROR(cudaMemcpy(device_img, host_img, (size*size)*sizeof(double), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(device_img_past, host_img_past, (size*size)*sizeof(double), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(device_img_future, host_img_future, (size*size)*sizeof(double), cudaMemcpyDeviceToHost));

        HANDLE_ERROR(cudaMemcpy(device_potential, host_potential, (size*size)*sizeof(double), cudaMemcpyDeviceToHost));

        time_dependence(host_real_past, host_real, host_real_future);
        time_dependence(host_img_past, host_img, host_img_future);

        for(i = 0; i < size; i++){
            for(j = 0; j < size; j++){
                host_psi[i * size + j] = host_real[i * size + j] + I*host_img[i * size + j];
            }
        }
    }

    archiving(host_psi);

    free(host_psi);
    free(host_real);
    free(host_real_past);
    free(host_real_future);
    free(host_img);
    free(host_img_past);
    free(host_img_future);
    free(host_potential);

    cudaFree(device_real);
    cudaFree(device_real_past);
    cudaFree(device_real_future);
    cudaFree(device_img);
    cudaFree(device_img_past);
    cudaFree(device_img_future);
    cudaFree(device_potential);

    clock_t endTime = clock();
 
	final_time = fopen("temponoflags.txt", "w");
	fprintf(final_time, "%10.2f", (endTime - beginTime)/(1.0*CLOCKS_PER_SEC));
	fclose(final_time);
	
    return 0;
}
