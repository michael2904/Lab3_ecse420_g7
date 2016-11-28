/* Example of using lodepng to load, process, save image */
#include <stdio.h>
#include <stdlib.h>
#define N 512 // grid side length
#define RHO 0.5 // related to pitch
#define ETA 2e-4 // related to duration of sound
#define BOUNDARY_GAIN 0.75 // clamped edge vs free edge


#define BLOCK_WIDTH 512

//Putting blocks of size width divided by 0, so that each thread can access the neighboring values. There is no neighboring value that is called twice.

__global__ void grid_N_First_Step(float u_out[N][N], float u1_in[N][N],float u2_in[N][N]){

	int ind = blockIdx.x * blockDim.x + threadIdx.x;
	int i = ((ind) / ((N)))+1;
	int j = ((ind) % (N))+1;
	if(i< N-1 && j<N-1){
		//do work
		// float sum_of_neighbors, previous_value, previous_previous_value;
		// sum_of_neighbors = u1_in[i-1][j] + u1_in[i+1][j] + u1_in[i][j-1] + u1_in[i][j+1];
		// previous_value = u1_in[i][j];
		// previous_previous_value = u2_in[i][j];
		// u_out[i][j] = (RHO * (sum_of_neighbors -4*previous_value) + 2*previous_value -(1-ETA)*previous_previous_value)/(1+ETA);
	}
}

__global__ void grid_N_Second_Step(float u_out[N][N], float u_in[N][N]){

	int ind = blockIdx.x * blockDim.x + threadIdx.x;
	int i = ((ind) / ((N)))+1;
	int j = ((ind) % (N))+1;
	if(i< N-1 && j == 0){
		// //do work
		// if(j == 0){
		// 	u_out[0][i] = BOUNDARY_GAIN * u_in[1][i]; // top
		// }else if(j == 1){
		// 	u_out[N-1][i] = BOUNDARY_GAIN * u_in[N-2][i]; // bottom
		// }else if(j == 2){
		// 	u_out[i][0] = BOUNDARY_GAIN * u_in[i][1]; // left
		// }else if(j == 3){
		// 	u_out[i][N-1] = BOUNDARY_GAIN * u_in[i][N-2]; // right
		// }
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
	float ** u1_in;
	float ** u2_in;
	float ** u_in;
	float ** u_out;
	float ** temp;


	int t;
	for (t = 0; t < T; t++) {
		printf("Run %d | %d total size with width %d and height %d in %d blocks of size %d. Size of memory %d\n",t,(N*N),N,N, ((N*N)+(BLOCK_WIDTH-1))/BLOCK_WIDTH, BLOCK_WIDTH,size);
		printf(" Try printing %f\n",u[N/2][N/2]);
		// allocate GPU memory
		cudaMalloc((void**) &u1_in, size);
		cudaError_t error0 = cudaGetLastError();
		printf("1st malloc: %s\n",cudaGetErrorString(error0));
		cudaMalloc((void**) &u2_in, size);
		error0 = cudaGetLastError();
		printf("2nd malloc: %s\n",cudaGetErrorString(error0));
		cudaMalloc((void**) &u_out, size);
		error0 = cudaGetLastError();
		printf("3rd malloc: %s\n",cudaGetErrorString(error0));

		// transfer the array to the GPU
		cudaMemcpy(u1_in, u1, size, cudaMemcpyHostToDevice);
		cudaMemcpy(u2_in, u2, size, cudaMemcpyHostToDevice);

		error0 = cudaGetLastError();
		printf("1st copy: %s\n",cudaGetErrorString(error0));

		// launch the kernel
		dim3 dimGrid(((N*N)+(BLOCK_WIDTH-1))/BLOCK_WIDTH);
		dim3 dimBlock(BLOCK_WIDTH);

		grid_N_First_Step<<<dimGrid, dimBlock>>>((float(*) [N])u_out,(float(*) [N]) u1_in,(float(*) [N])u2_in);
		
		error0 = cudaGetLastError();
		printf("1st launch: %s\n",cudaGetErrorString(error0));

		// copy back the result array to the CPU
		cudaMemcpy(u, u_out, size, cudaMemcpyDeviceToHost);

		error0 = cudaGetLastError();
		printf("2n copy: %s\n",cudaGetErrorString(error0));

		cudaFree(u2_in);
		cudaFree(u1_in);
		cudaFree(u_out);

		cudaError_t error1 = cudaGetLastError();
		printf("kernel 1 launch failed: %s\n",cudaGetErrorString(error1));


		cudaThreadSynchronize();

		cudaError_t error2 = cudaGetLastError();
		printf("kernel 1 execution failed: %s\n",cudaGetErrorString(error2));

		// second step

		// allocate GPU memory
		cudaMalloc(&u_in, size);
		error0 = cudaGetLastError();
		printf("3rd malloc: %s\n",cudaGetErrorString(error0));
		cudaMalloc(&u_out, size);
		error0 = cudaGetLastError();
		printf("4 malloc: %s\n",cudaGetErrorString(error0));
		// transfer the array to the GPU
		cudaMemcpy(u_in, u, size, cudaMemcpyHostToDevice);
		error0 = cudaGetLastError();
		printf("5 malloc: %s\n",cudaGetErrorString(error0));

		// // launch the kernel
		// dim3 dimGrid((size+(BLOCK_WIDTH-1))/BLOCK_WIDTH);
		// dim3 dimBlock(BLOCK_WIDTH);

		grid_N_Second_Step<<<dimGrid, dimBlock>>>((float(*) [N])u_out,(float(*) [N]) u_in);
		error0 = cudaGetLastError();
		printf("6 running: %s\n",cudaGetErrorString(error0));
		// copy back the result array to the CPU
		cudaMemcpy(u, u_out, size, cudaMemcpyDeviceToHost);
		error0 = cudaGetLastError();
		printf("7 copy: %s\n",cudaGetErrorString(error0));

		cudaFree(u_in);
		error0 = cudaGetLastError();
		printf("8 free: %s\n",cudaGetErrorString(error0));
		cudaFree(u_out);
		error0 = cudaGetLastError();
		printf("9 free: %s\n",cudaGetErrorString(error0));
		cudaError_t error3 = cudaGetLastError();
		printf("kernel 2 launch failed: %s\n",cudaGetErrorString(error3));

		cudaThreadSynchronize();
		cudaError_t error4 = cudaGetLastError();
		printf("kernel 2 execution failed: %s\n",cudaGetErrorString(error4));
		printf("Try 162 printing %f %f %f %f \n",u[N/2][N/2],u1[N/2][N/2],u2[N/2][N/2],u[N/2][N/2]);

		printf(" Try 99 printing %f\n",u[N/2][N/2]);
		// update corners
		u[0][0] = BOUNDARY_GAIN * u[1][0];
		error0 = cudaGetLastError();
		printf("10 copy: %s\n",cudaGetErrorString(error0));
		u[N-1][0] = BOUNDARY_GAIN * u[N-2][0];
		error0 = cudaGetLastError();
		printf("11 copy: %s\n",cudaGetErrorString(error0));
		u[0][N-1] = BOUNDARY_GAIN * u[0][N-2];
		error0 = cudaGetLastError();
		printf("12 copy: %s\n",cudaGetErrorString(error0));
		u[N-1][N-1] = BOUNDARY_GAIN * u[N-1][N-2];
		error0 = cudaGetLastError();
		printf("13 copy: %s\n",cudaGetErrorString(error0));

		// print_grid(u);

		audio[t] = u[N/2][N/2];
		error0 = cudaGetLastError();
		printf("14 copy: %s\n",cudaGetErrorString(error0));
		printf("%f,\n", audio[t]);
		printf("Try 15 printing %f\n",u[N*N/2]);
		temp = u2;
		printf("Try 161 printing %f %f %f %f \n",u2[N/2][N/2],u2[N/2][N/2],u2[N/2][N/2],temp[N/2][N/2]);
		u2 = u1;
		printf("Try 162 printing %f %f %f %f \n",u[N/2][N/2],u1[N/2][N/2],u2[N/2][N/2],temp[N/2][N/2]);
		u1 = u;
		printf("Try 163 printing %f %f %f %f \n",u[N/2][N/2],u1[N/2][N/2],u2[N/2][N/2],temp[N/2][N/2]);
		u = temp;
		printf("Try 164 printing %f %f %f %f \n",u[N/2][N/2],u1[N/2][N/2],u2[N/2][N/2],temp[N/2][N/2]);
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
