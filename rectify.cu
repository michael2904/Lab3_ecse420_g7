/* Example of using lodepng to load, process, save image */
#include "lodepng.h"
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_WIDTH 1000
#define MAX_MSE 0.00001f


__global__ void rectify(unsigned char * d_out, unsigned char * d_in){
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	unsigned char f = d_in[idx];
	if(idx % 4 != 3){
		if (f < 127){
			f = 127;
		}
		f = f < 127 ? 127 : f;
	}
	d_out[idx] = f;
}


int process(char* input_filename, char* output_filename){
	unsigned error;
	unsigned char *image, *new_image;
	unsigned width, height;

	//image --> h_in
	//new_image --> h_out

	error = lodepng_decode32_file(&image, &width, &height, input_filename);
	if(error){
		printf("error %u: %s\n", error, lodepng_error_text(error));
		return error;
	}

	const int size = width * height * 4 * sizeof(unsigned char);
	new_image = (unsigned char *)malloc(size);


	// declare GPU memory pointers
	unsigned char * d_in;
	unsigned char * d_out;

	// allocate GPU memory
	cudaMalloc(&d_in, size);
	cudaMalloc(&d_out, size);

	// transfer the array to the GPU
	cudaMemcpy(d_in, image, size, cudaMemcpyHostToDevice);

	printf("%d total threads in %d blocks of size %d\n",size, (size/BLOCK_WIDTH + (size % BLOCK_WIDTH > 0)), BLOCK_WIDTH);

	// launch the kernel
	rectify<<<(size/BLOCK_WIDTH + (size % BLOCK_WIDTH > 0)), BLOCK_WIDTH>>>(d_out, d_in);

	// copy back the result array to the CPU
	cudaMemcpy(new_image, d_out, size, cudaMemcpyDeviceToHost);

	cudaFree(d_in);
	cudaFree(d_out);

	lodepng_encode32_file(output_filename, new_image, width, height);

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
