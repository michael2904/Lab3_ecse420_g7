/* Example of using lodepng to load, process, save image */
#include "lodepng.h"
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_WIDTH 1000


__global__ void rectify(unsigned char * d_out, unsigned char * d_in){
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	unsigned char f = d_in[idx];
	if(idx <1000){
		printf("thread %d in block %d: idx = %d and f is %d\n", threadIdx.x, blockIdx.x, idx,f);
	}
	//if(idx % 4 != 3){
		//f = f < 127 ? 127 : f; // R
	//}
	d_out[idx] = f;
	if(idx <1000 ){
		printf("thread %d in block %d: idx = %d and became %d\n", threadIdx.x, blockIdx.x, idx,d_out[idx]);
	}
}


int process(char* input_filename, char* output_filename)
{
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
	int i;
	for(i = 0; i<1000;i++){
		printf("This is image at %d : %d\n",i,image[i]);
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

	// launch the kernel
	rectify<<<size/BLOCK_WIDTH, BLOCK_WIDTH>>>(d_out, d_in);

	// copy back the result array to the CPU
	cudaMemcpy(new_image, d_out, size, cudaMemcpyDeviceToHost);

	// // rectify image
	// unsigned char value;
	// int i,j;
	// for (i = 0; i < height; i++) {
	// 	for (j = 0; j < width; j++) {
	// 		for (int k = 0; k < 3; k++) {
	// 			new_image[4*width*i + 4*j + k] = image[4*width*i + 4*j + k] < 127 ? 127 : image[4*width*i + 4*j + k]; // R
	// 		}
	// 		new_image[4*width*i + 4*j + 3] = image[4*width*i + 4*j + 3]; // A
	// 	}
	// }

	cudaFree(d_in);
	cudaFree(d_out);
	int j;
	for(j = 0; j<1000;j++){
		printf("This was image at %d: %d and now it is: %d\n",j,image[j],new_image[j]);
	}
	lodepng_encode32_file(output_filename, new_image, width, height);

	free(image);
	free(new_image);
	return 0;
}

int main(int argc, char *argv[])
{
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
