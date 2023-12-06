#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include <cuda_runtime.h>

//break program into two parallel functions, one to do pairwise and one to sum

__global__ void pairwise(vector3 *hPos, vector3 *accels, double *mass){
        int column = (blockDim.x * blockIdx.x) + threadIdx.x;
        int row = (blockDim.y * blockIdx.y) + threadIdx.y;
        int index = (NUMENTITIES * row) + column;
        if(index < NUMENTITIES * NUMENTITIES){
                if (row == column){
                        FILL_VECTOR(accels[index], 0, 0, 0);
                } else{
			vector3 distance;
			for (int k=0;k<3;k++) distance[k]=hPos[row][k]-hPos[column][k];
			double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
			double magnitude=sqrt(magnitude_sq);
			double accelmag=-1*GRAV_CONSTANT*mass[column]/magnitude_sq;
			FILL_VECTOR(accels[index],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);

                }
        }
}
__global__ void sum(vector3 *accels, vector3 *accel_sum, vector3 *hPos, vector3 *hVel){
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < NUMENTITIES){
		FILL_VECTOR(accel_sum[row], 0, 0, 0);
		for (int j = 0; j < NUMENTITIES; j++){
			for (int k = 0; k < 3; k++){
				accel_sum[row][k] += accels[(row * NUMENTITIES) + j][k];}
		}
		for (int k = 0; k < 3; k++){
			hVel[row][k] += accel_sum[row][k] * INTERVAL;
			hPos[row][k] = hVel[row][k] * INTERVAL;
		}

	}
}
void compute(){
	vector3 *dhPos, *dhVel;
	double *dMass;
	vector3 *dAccel, *dSum;
	int block = NUMENTITIES / 16.0f;
	int thread = NUMENTITIES / (float) block;
	dim3 gridDim(block, block, 1);
	dim3 blockDim(thread, thread, 1);
	
	cudaMallocManaged((void**) &dhPos, sizeof(vector3) * NUMENTITIES);
	cudaMallocManaged((void**) &dhVel, sizeof(vector3) * NUMENTITIES);
	cudaMallocManaged((void**) &dMass, sizeof(double) * NUMENTITIES);

	cudaMemcpy(dhPos, hPos, sizeof(vector3)*NUMENTITIES, cudaMemcpyHostToDevice);
	cudaMemcpy(dhVel, hVel, sizeof(vector3)*NUMENTITIES, cudaMemcpyHostToDevice);
	cudaMemcpy(dMass, mass, sizeof(double)*NUMENTITIES, cudaMemcpyHostToDevice);
	
	cudaMallocManaged((void**) &dAccel, sizeof(vector3) * NUMENTITIES);
        cudaMallocManaged((void**) &dSum, sizeof(vector3) * NUMENTITIES);

	pairwise<<<gridDim, blockDim>>>(dhPos, dAccel, dMass);
	cudaDeviceSynchronize();

	sum<<<gridDim.x, blockDim.x>>>(dAccel, dSum, dhPos, dhVel);
	cudaDeviceSynchronize();

	cudaMemcpy(hPos, dhPos, sizeof(vector3)*NUMENTITIES, cudaMemcpyDeviceToHost);
	cudaMemcpy(hVel, dhVel, sizeof(vector3)*NUMENTITIES, cudaMemcpyDeviceToHost);
	cudaMemcpy(dMass, mass, sizeof(double)*NUMENTITIES, cudaMemcpyHostToDevice);

	cudaFree(dhPos);
	cudaFree(dhVel);
	cudaFree(dAccel);
	cudaFree(dMass);
}
