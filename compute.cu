#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include <cuda_runtime.h>

//break program into two parallel functions, one to do pairwise and one to sum

__global__ void pairwise(vector3 *hPos, vector3 *accels, double *mass){
        int row = (blockDim.x * blockIdx.x) + threadIdx.x;
        int column = (blockDim.y * blockIdx.y) + threadIdx.y;
        int index = (NUMENTITIES * row) + column;
        if(row >= NUMENTITIES || column >= NUMENTITIES)return;
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
__global__ void sum(vector3 *accels, vector3 *hPos, vector3 *hVel){
	int row = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(row > NUMENTITIES) return;
		vector3 accel_sum = {0,0,0};
		for(int j = 0; j < NUMENTITIES; j++){
			for (int k = 0; k < 3; k++){
				accel_sum[k] += accels[row * NUMENTITIES + j][k];}}
		for (int k = 0; k < 3; k++){
			hVel[row][k] += accel_sum[k] * INTERVAL;
			hPos[row][k] = hVel[row][k] * INTERVAL;
		}
}
void compute(vector3* dAccel, vector3* dhPos, vector3* dhVel, double* dMass){
	
	

	dim3 dimBlock(16, 16);
	dim3 dimGrid((NUMENTITIES+dimBlock.x-1)/dimBlock.x, (NUMENTITIES+dimBlock.y-1)/dimBlock.y);
	pairwise<<<dimGrid, dimBlock>>>(dhPos, dAccel, dMass);
	cudaDeviceSynchronize();
	
	dim3 dimBlock2(256);
	dim3 gridDim2((NUMENTITIES+dimBlock.x-1)/dimBlock.x);	
	sum<<<gridDim2, dimBlock2>>>(dAccel, dhPos, dhVel);
	cudaDeviceSynchronize();

}
