/* Example of using lodepng to load, process, save image */
#include <stdio.h>
#include <stdlib.h>
#define N 512 // grid side length
#define RHO 0.5 // related to pitch
#define ETA 2e-4 // related to duration of sound
#define BOUNDARY_GAIN 0.75 // clamped edge vs free edge


#define BLOCK_WIDTH 1000

//Putting blocks of size width divided by 0, so that each thread can access the neighboring values. There is no neighboring value that is called twice.

__global__ void grid_N_First_Step(float u_out[N][N], float u1_in[N][N],float u2_in[N][N]){

	int ind = blockIdx.x * blockDim.x + threadIdx.x;
	int i = ((ind) / ((N)))+1;
	int j = ((ind) % (N))+1;
	if(i< N-1 && j<N-1){
		//do work
		float sum_of_neighbors, previous_value, previous_previous_value;
		sum_of_neighbors = u1_in[i-1][j] + u1_in[i+1][j] + u1_in[i][j-1] + u1_in[i][j+1];
		previous_value = u1_in[i][j];
		previous_previous_value = u2_in[i][j];
		u_out[i][j] = (RHO * (sum_of_neighbors -4*previous_value) + 2*previous_value -(1-ETA)*previous_previous_value)/(1+ETA);
	}
}

__global__ void grid_N_Second_Step(float u_out[N][N], float u_in[N][N]){

	int ind = blockIdx.x * blockDim.x + threadIdx.x;
	int i = ((ind) / ((N)))+1;
	int j = ((ind) % (N))+1;
	if(i< N-1 && j == 0){
		//do work
		if(j == 0){
			u_out[0][i] = BOUNDARY_GAIN * u_in[1][i]; // top
		}else if(j == 1){
			u_out[N-1][i] = BOUNDARY_GAIN * u_in[N-2][i]; // bottom
		}else if(j == 2){
			u_out[i][0] = BOUNDARY_GAIN * u_in[i][1]; // left
		}else if(j == 3){
			u_out[i][N-1] = BOUNDARY_GAIN * u_in[i][N-2]; // right
		}
	}
}


int process(int T){
	// initialize grid
	float **u = (float **) malloc(N * sizeof(float *));
	float **u1 = (float **) malloc(N * sizeof(float *));
	float **u2 = (float **) malloc(N * sizeof(float *));
	int i,j;
	for (i = 0; i < N; i++) {
		u[i] = (float *)malloc(N * sizeof(float)); 
		u1[i] = (float *)malloc(N * sizeof(float));
		u2[i] = (float *)malloc(N * sizeof(float));
		for (j = 0; j < N; j++) {
			u[i][j] = 0;
			u1[i][j] = 0;
			u2[i][j] = 0;
		}
	}
	printf("Size of grid: %d nodes\n", N*N);
	// simulate drum strike
	u1[N/2][N/2] = 1;
	float *audio = (float *) malloc(T * sizeof(float));
	const int size = N * N * sizeof(float);

	// declare GPU memory pointers
	float * u1_in;
	float * u2_in;
	float * u_in;
	float * u_out;

	float **temp;


	int t;
	for (t = 0; t < T; t++) {

		// allocate GPU memory
		cudaMalloc(&u1_in, size);
		cudaMalloc(&u2_in, size);

		// transfer the array to the GPU
		cudaMemcpy(u1_in, u1, size, cudaMemcpyHostToDevice);
		cudaMemcpy(u2_in, u2, size, cudaMemcpyHostToDevice);

		printf("Run %d | %d total size with width %d and height %d in %d blocks of size %d\n",t,size,N,N, (size+(BLOCK_WIDTH-1))/BLOCK_WIDTH, BLOCK_WIDTH);

		// launch the kernel
		dim3 dimGrid((size+(BLOCK_WIDTH-1))/BLOCK_WIDTH);
		dim3 dimBlock(BLOCK_WIDTH);

		grid_N_First_Step<<<dimGrid, dimBlock>>>((float(*) [N])u_out,(float(*) [N]) u1_in,(float(*) [N])u2_in);

		// copy back the result array to the CPU
		cudaMemcpy(u, u_out, size, cudaMemcpyDeviceToHost);

		cudaFree(u2_in);
		cudaFree(u1_in);
		cudaFree(u_out);

		if (cudaGetLastError() != cudaSuccess) printf("kernel 1 launch failed\n");

		cudaThreadSynchronize();

		if (cudaGetLastError() != cudaSuccess) printf("kernel 1 execution failed\n");

		// second step

		// allocate GPU memory
		cudaMalloc(&u_in, size);

		// transfer the array to the GPU
		cudaMemcpy(u_in, u, size, cudaMemcpyHostToDevice);

		// // launch the kernel
		// dim3 dimGrid((size+(BLOCK_WIDTH-1))/BLOCK_WIDTH);
		// dim3 dimBlock(BLOCK_WIDTH);

		grid_N_Second_Step<<<dimGrid, dimBlock>>>((float(*) [N])u_out,(float(*) [N]) u_in);

		// copy back the result array to the CPU
		cudaMemcpy(u, u_out, size, cudaMemcpyDeviceToHost);

		cudaFree(u_in);
		cudaFree(u_out);

		if (cudaGetLastError() != cudaSuccess) printf("kernel 1 launch failed\n");

		cudaThreadSynchronize();

		if (cudaGetLastError() != cudaSuccess) printf("kernel 1 execution failed\n");


		// update corners
		u[0][0] = BOUNDARY_GAIN * u[1][0];
		u[N-1][0] = BOUNDARY_GAIN * u[N-2][0];
		u[0][N-1] = BOUNDARY_GAIN * u[0][N-2];
		u[N-1][N-1] = BOUNDARY_GAIN * u[N-1][N-2];

		// print_grid(u);

		audio[t] = u[N/2][N/2];
		printf("%f,\n", audio[t]);

		temp = u2;
		u2 = u1;
		u1 = u;
		u = temp;

		// record displacement at node (N-1,N-1)
	}

	// free grid memory
	for (i = 0; i < N; i++) {
		free(u[i]);
		free(u1[i]);
		free(u2[i]);
	}

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
