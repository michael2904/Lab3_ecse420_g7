/* Example of using lodepng to load, process, save image */
#include "lodepng.h"
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_WIDTH 1024

//Putting blocks of size width divided by 0, so that each thread can access the neighboring values. There is no neighboring value that is called twice.

__global__ void pool(unsigned char * d_out, unsigned char * d_in,int width,int height){

	int ind = blockIdx.x * blockDim.x + threadIdx.x;
	int i = (ind) / (width*4);
	int j = (ind/4) % (width);
	int k = (ind) % 4;
	int size = width * height * 4;
	if(i%2 == 0 && j%2 == 0 && k != 3 && ind < size){
		unsigned char max;
		max = d_in[4*width*i + 4*j + k];
		if(d_in[4*width*(i+1) + 4*j + k]>max) max = d_in[4*width*(i+1) + 4*j + k];
		if(d_in[4*width*(i+1) + 4*(j+1) + k]>max) max = d_in[4*width*(i+1) + 4*(j+1) + k];
		if(d_in[4*width*i + 4*(j+1) + k]>max) max = d_in[4*width*i + 4*(j+1) + k];
		d_out[width*i + j*2 + k] = max;
	}
	if(i%2 == 0 && j % 2 == 0 && k == 3 && ind < size){
		d_out[width * i + j*2 + 3] = d_in[4*width*i + 4*j + 3];
	}
}


int process(char* input_filename, char* output_filename){
	unsigned error;
	unsigned char *image, *new_image;
	unsigned width, height;
	unsigned new_width, new_height;

	error = lodepng_decode32_file(&image, &width, &height, input_filename);
	if(error){
		printf("error %u: %s\n", error, lodepng_error_text(error));
		return error;
	}
	new_width = (width)/2;
	new_height = (height)/2;

	const int size = width * height * 4 * sizeof(unsigned char);
	const int new_size = new_width * new_height * 4 * sizeof(unsigned char);

	new_image = (unsigned char *)malloc(new_size);


	// declare GPU memory pointers
	unsigned char * d_in;
	unsigned char * d_out;

	// allocate GPU memory
	cudaMalloc(&d_in, size);
	cudaMalloc(&d_out, new_size);

	// transfer the array to the GPU
	cudaMemcpy(d_in, image, size, cudaMemcpyHostToDevice);

	printf("%d total size with width %d and height %d in %d blocks of size %d\n",size,width,height, (size+(BLOCK_WIDTH-1))/BLOCK_WIDTH, BLOCK_WIDTH);

	// launch the kernel
	dim3 dimGrid((size+(BLOCK_WIDTH-1))/BLOCK_WIDTH);
	dim3 dimBlock(BLOCK_WIDTH);


	pool<<<dimGrid, dimBlock>>>(d_out, d_in,width,height);

	// copy back the result array to the CPU
	cudaMemcpy(new_image, d_out, new_size, cudaMemcpyDeviceToHost);

	cudaFree(d_in);
	cudaFree(d_out);

	lodepng_encode32_file(output_filename, new_image, new_width, new_height);
	int i,j,k;
	for(i = 0; i<8;i++){
		for(j = 0;j<8;j++){
			printf("(%d,%d,%d):",i,j,k);
			printf(":%d",image[4*width*i + 4*j + 0]);
			printf(" | ");
		}
		printf("\n");
	}
	for(i = 0; i<4;i++){
		for(j = 0;j<4;j++){
			printf("(%d,%d,%d):",i,j,k);
			printf(":%d",new_image[4*new_width*i + 4*j + 0]);
			printf(" | ");
		}
		printf("\n");
	}

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
