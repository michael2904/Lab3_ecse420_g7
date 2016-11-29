/* Example of using lodepng to load, process, save image */
#include <stdio.h>
#include <stdlib.h>
#define N 4 // grid side length
#define RHO 0.5 // related to pitch
#define ETA 2e-4 // related to duration of sound
#define BOUNDARY_GAIN 0.75 // clamped edge vs free edge


#define BLOCK_WIDTH 512
#define ind(i,j) ((j) + ((i)*(N)))

//Putting blocks of size width divided by 0, so that each thread can access the neighboring values. There is no neighboring value that is called twice.

__global__ void grid_N(float * u_out, float * u1_in,float * u2_in){

	int ind = blockIdx.x * blockDim.x + threadIdx.x;
	int i = ((ind) / ((N))) + 1;
	int j = ((ind) % (N)) + 1;
	if(i<N && j < N){
		printf("%d u(%d,%d) %f \n",ind,i,j,u_out[ind(i,j)]);
		__syncthreads();
		if(i< N-1 && j<N-1){
			//do work
			float sum_of_neighbors, previous_value, previous_previous_value;
			sum_of_neighbors = u1_in[ind(i-1,j)] + u1_in[ind(i+1,j)] + u1_in[ind(i,j-1)] + u1_in[ind(i,j+1)];
			previous_value = u1_in[ind(i,j)];
			previous_previous_value = u2_in[ind(i,j)];
			u_out[ind(i,j)] = (RHO * (sum_of_neighbors -4*previous_value) + 2*previous_value -(1-ETA)*previous_previous_value)/(1+ETA);
		}
		__syncthreads();
		printf("%d u(%d,%d) %f \n",ind,i,j,u_out[ind(i,j)]);
		__syncthreads();
		if(i< N-1 && j == 0){
			//do work
			if(j == 0){
				__syncthreads();
				printf("--- %d u(%d,%d) %f  = %f\n",ind,i,j,u_out[ind(0,i)],BOUNDARY_GAIN * u_out[ind(1,i)]);
				__syncthreads();
				u_out[ind(0,i)] = BOUNDARY_GAIN * u_out[ind(1,i)]; // top
				__syncthreads();
				printf("--- %d u(%d,%d) %f \n",ind,i,j,u_out[ind(0,i)]);
				__syncthreads();
			}else if(j == 1){
				u_out[ind(N-1,i)] = BOUNDARY_GAIN * u_out[ind(N-2,i)]; // bottom
			}else if(j == 2){
				u_out[ind(i,0)] = BOUNDARY_GAIN * u_out[ind(i,1)]; // left
			}else if(j == 3){
				u_out[ind(i,N-1)] = BOUNDARY_GAIN * u_out[ind(i,N-2)]; // right
			}
		}
		__syncthreads();
		printf("%d u(%d,%d) %f \n",ind,i,j,u_out[ind(i,j)]);
		__syncthreads();
		if(j == 0){
			// update corners
			if(i == 0){
				u_out[ind(0,0)] = BOUNDARY_GAIN * u_out[ind(1,0)];
			}else if(i == 1){
				u_out[ind(N-1,0)] = BOUNDARY_GAIN * u_out[ind(N-2,0)];
			}else if(i == 2){
				u_out[ind(0,N-1)] = BOUNDARY_GAIN * u_out[ind(0,N-2)];
			}else if(i == 3){
				u_out[ind(N-1,N-1)] = BOUNDARY_GAIN * u_out[ind(N-1,N-2)];
			}
		}
		__syncthreads();
		printf("%d u(%d,%d) %f \n",ind,i,j,u_out[ind(i,j)]);
	}
	i = i -1;
	j = j -1;
	__syncthreads();
	if(i<N && j < N){
		printf("%d u(%d,%d) %f \n",ind,i,j,u_out[ind(i,j)]);
	}
	__syncthreads();

}



int process(int T){
	// initialize grid
	float *u = (float *) malloc(N * N * sizeof(float *));
	float *u1 = (float *) malloc(N * N * sizeof(float *));
	float *u2 = (float *) malloc(N * N * sizeof(float *));
	int i,j;
	for (i = 0; i < N*N; i++) {
		u[i] = 0;
		u1[i] = 0;
		u2[i] = 0;
	}
	printf("Size of grid: %d nodes\n", N*N);
	// simulate drum strike
	u1[ind(N/2,N/2)] = 1;
	float *audio = (float *) malloc(T * sizeof(float));
	const int size = N * N * sizeof(float);

	// declare GPU memory pointers
	float * u1_in;
	float * u2_in;
	float * u_out;
	float * temp;

	// allocate GPU memory
	cudaMalloc(&u1_in, size);
	cudaMalloc(&u2_in, size);
	cudaMalloc(&u_out, size);

	int t;
	for (t = 0; t < T; t++) {
		// printf("Run %d | %d total size with width %d and height %d in %d blocks of size %d. Size of memory %d\n",t,(N*N),N,N, ((N*N)+(BLOCK_WIDTH-1))/BLOCK_WIDTH, BLOCK_WIDTH,size);
		printf("Try printing %f %f %f\n",u[ind(N/2,N/2)],u1[ind(N/2,N/2)],u2[ind(N/2,N/2)]);
		for (i = 0; i < N; i++) {
			for (j = 0; j < N; j++) {
				printf("u(%d,%d) %f |",i,j,u[ind(i,j)]);
			}
			printf("\n");
		}
		printf("\n");
		for (i = 0; i < N; i++) {
			for (j = 0; j < N; j++) {
				printf("u1(%d,%d) %f |",i,j,u1[ind(i,j)]);
			}
			printf("\n");
		}
		printf("\n");
		for (i = 0; i < N; i++) {
			for (j = 0; j < N; j++) {
				printf("u2(%d,%d) %f |",i,j,u2[ind(i,j)]);
			}
			printf("\n");
		}


		// transfer the array to the GPU
		cudaMemcpy(u1_in, u1, size, cudaMemcpyHostToDevice);
		cudaMemcpy(u2_in, u2, size, cudaMemcpyHostToDevice);

		// launch the kernel
		dim3 dimGrid(((N*N)+(BLOCK_WIDTH-1))/BLOCK_WIDTH);
		dim3 dimBlock(BLOCK_WIDTH);

		grid_N<<<dimGrid, dimBlock>>>(u_out,u1_in,u2_in);

		// copy back the result array to the CPU
		cudaMemcpy(u, u_out, size, cudaMemcpyDeviceToHost);

		cudaError_t error1 = cudaGetLastError();
		if (error1 != cudaSuccess)printf("kernel 1 launch failed: %s\n",cudaGetErrorString(error1));
		cudaThreadSynchronize();
		cudaError_t error2 = cudaGetLastError();
		if (error2 != cudaSuccess)printf("kernel 1 execution failed: %s\n",cudaGetErrorString(error2));


		// print_grid(u);

		audio[t] = u[ind(N/2,N/2)];
		printf("%f,\n", audio[t]);
		temp = u2;
		u2 = u1;
		u1 = u;
		u = temp;
	}

	cudaFree(u2_in);
	cudaFree(u1_in);
	cudaFree(u_out);

	free(u);
	free(u1);
	free(u2);
	free(audio);
	return 0;
}

int main(int argc, char *argv[]){
	if ( argc >= 2 ){
		int T = atoi(argv[1]);

		int error = process(T);

		if(error != 0){
			printf("An error occured. ( %d )\n",error);

		}else{
			printf("The rectification ran with success.\n");
		}
	}else{
		printf("There is inputs missing.\n");
	}
	return 0;
}
