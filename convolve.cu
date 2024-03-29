/* Example of using lodepng to load, process, save image */
#include "lodepng.h"
#include <stdio.h>
#include <stdlib.h>
#include "wm.h"
#include <sys/time.h>

#define BLOCK_WIDTH 1024

//Putting blocks of size width divided by 0, so that each thread can access the neighboring values. There is no neighboring value that is called twice.

__global__ void convolve(unsigned char * d_out, unsigned char * d_in,int width,int height,float w[3][3]){

	int ind = blockIdx.x * blockDim.x + threadIdx.x;
	int i = ((ind) / ((width)*4))+1;
	int j = ((ind/4) % (width))+1;
	int k = (ind) % 4;
	if(i< height-1 && j<width-1){
		if(k != 3){
			float currentWF = 0;
			float value = 0;
			int ii,jj;
			for (ii = 0; ii < 3; ii++) {
				for (jj = 0; jj < 3; jj++) {
					currentWF = w[ii][jj];
					value += ((float) d_in[4*(width)*(i+ii-1) + 4*(j+jj-1) + k]) * currentWF;
				}
			}
			if((value)<0) value = 0;
			if((value)>255) value = 255;
			d_out[4*(width-2)*(i-1) + 4*(j-1) + k] = (unsigned char) value;
		}else if( k == 3){
			d_out[4*(width-2)*(i-1) + 4*(j-1) + 3] = 255;
		}
	}
}


int process(char* input_filename, char* output_filename){
	unsigned error;
	unsigned char *image, *new_image;
	unsigned width, height;
	unsigned new_width, new_height;

	struct timeval start_time, end_time;


	error = lodepng_decode32_file(&image, &width, &height, input_filename);
	if(error){
		printf("error %u: %s\n", error, lodepng_error_text(error));
		return error;
	}


	gettimeofday(&start_time, NULL);
	
	new_width = (width)-2;
	new_height = (height)-2;

	const int size = width * height * 4 * sizeof(unsigned char);
	const int new_size = new_width * new_height * 4 * sizeof(unsigned char);

	new_image = (unsigned char *)malloc(new_size);


	// declare GPU memory pointers
	unsigned char * d_in;
	unsigned char * d_out;
	float *w_d;

	// allocate GPU memory
	cudaMalloc(&d_in, size);
	cudaMalloc(&d_out, new_size);
	cudaMalloc((void**) &w_d, 9 * sizeof(float));

	// transfer the array to the GPU
	cudaMemcpy(d_in, image, size, cudaMemcpyHostToDevice);
	cudaMemcpy (w_d, w, 9 * sizeof(float), cudaMemcpyHostToDevice);


	printf("%d total size with width %d and height %d in %d blocks of size %d\n",new_size,new_width,new_height, (new_size+(BLOCK_WIDTH-1))/BLOCK_WIDTH, BLOCK_WIDTH);

	// launch the kernel
	dim3 dimGrid((size+(BLOCK_WIDTH-1))/BLOCK_WIDTH);
	dim3 dimBlock(BLOCK_WIDTH);


	convolve<<<dimGrid, dimBlock>>>(d_out, d_in,width,height,(float(*) [3])w_d);


	gettimeofday(&end_time, NULL);


	// copy back the result array to the CPU
	cudaMemcpy(new_image, d_out, new_size, cudaMemcpyDeviceToHost);

	cudaFree(d_in);
	cudaFree(d_out);
	cudaFree(w_d);

	if (cudaGetLastError() != cudaSuccess) printf("kernel launch failed\n");

	cudaThreadSynchronize();

	if (cudaGetLastError() != cudaSuccess) printf("kernel execution failed\n");

	lodepng_encode32_file(output_filename, new_image, new_width, new_height);


	  unsigned long long time_elapsed = 1000 * (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_usec - start_time.tv_usec) / 1000;
 
  printf("Time Elapsed [%llu ms]\n", time_elapsed);

	free(image);
	free(new_image);
	return 0;
}

int main(int argc, char *argv[]){
	if ( argc >= 3 ){
		char* input_filename = argv[1];
		char* output_filename = argv[2];

		int error = process(input_filename, output_filename);

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
