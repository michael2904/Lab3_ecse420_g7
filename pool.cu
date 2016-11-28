/* Example of using lodepng to load, process, save image */
#include "lodepng.h"
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_WIDTH 512

//Putting blocks of size width divided by 0, so that each thread can access the neighboring values. There is no neighboring value that is called twice.

__global__ void pool(unsigned char * d_out, unsigned char * d_in,int width,int height){

	int ind = blockIdx.x * blockDim.x + threadIdx.x;
	int i = (ind + width - 1) / width;
	int j = (ind + width - 1) % width;
	int k = (ind + 3) % 4;
	int size = width * height * 4;

	if(ind<2000) {
		printf("ind: %05d and width is %d height is %d : (%d,%d,%d)\n",ind,width,height,i,j,k);
	}
	if((j % 100 == 0 )&&(i % 100 == 0) && ind < size){
		printf("Original ind: %010d at (%04d,%04d,%04d)\n",ind,i,j,k);
	}


	//unsigned char max;
	// int new_width = (width+1)/2;
	// if(j%2 == 0 && k != 3){
		//max = d_in[4*width*i + 4*j + k];
		//if(d_in[4*width*(i+1) + 4*j + k]>max) max = d_in[4*width*(i+1) + 4*j + k];
		//if(d_in[4*width*(i+1) + 4*(j+1) + k]>max) max = d_in[4*width*(i+1) + 4*(j+1) + k];
		//if(d_in[4*width*i + 4*(j+1) + k]>max) max = d_in[4*width*i + 4*(j+1) + k];
		// d_out[new_width*i + j*2 + k] = max;
		// if(j < 10 )printf("Original max at (%d,%d,%d) for ind = %d\n",i,j,k,ind);
	// }
	// if(j % 2 == 0 && k == 3){
		//d_out[new_width * i + j*2 + 3] = d_in[4*width*i + 4*j + 3];
	// 	if(j < 10 )printf("Original max at (%d,%d,%d) for ind = %d\n",i,j,k,ind);
	// }
}


int process(char* input_filename, char* output_filename){
	unsigned error;
	unsigned char *image, *new_image;
	unsigned width, height;
	unsigned new_width, new_height;

	//image --> h_in
	//new_image --> h_out

	error = lodepng_decode32_file(&image, &width, &height, input_filename);
	if(error){
		printf("error %u: %s\n", error, lodepng_error_text(error));
		return error;
	}
	new_width = (width+1)/2;
	new_height = (height+1)/2;

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

	//lodepng_encode32_file(output_filename, new_image, width, height);
	//int i;
	//for(i = 0; i<128;i++)printf("new_image[%d] = %d\n",i,new_image[i]);

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
