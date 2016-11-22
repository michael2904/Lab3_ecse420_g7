/* Example of using lodepng to load, process, save image */
#include "lodepng.h"
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_WIDTH 512

//Putting blocks of size width divided by 0, so that each thread can access the neighboring values. There is no neighboring value that is called twice.

__global__ void pool(unsigned char * d_out, unsigned char * d_in){
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	int jdx = blockDim.y*blockIdx.y + threadIdx.y;
	int kdx = blockDim.z*blockIdx.z + threadIdx.z;
	int index = blockIdx.x * blockDim.x * blockDim.y * blockDim.z + threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
	int width = blockDim.x*blockDim.y*blockDim.z;
	//int i = index /width;
	//int j = (index / 4) % (width/4);
	//int k = index % 4;

	if(blockIdx.x == 0){
		printf("thread %d in block %d: index = %d so (%d,%d,%d)\n", threadIdx.x, blockIdx.x, index,idx,jdx,kdx);
	}

	//unsigned char max;
	//int new_width = width/2;
	//if(jdx%2 == 0 && kdx != 3){
	//	max = d_in[4*width*idx + 4*jdx + kdx];
	//	if(blockIdx.x == 0)printf("Original max = %d at (%d,%d,%d) for index = %d\n",max,idx,jdx,kdx,index);
	//	if(d_in[4*width*(idx+1) + 4*jdx + kdx]>max) max = d_in[4*width*(idx+1) + 4*jdx + kdx];
	//	if(d_in[4*width*(idx+1) + 4*(jdx+1) + kdx]>max) max = d_in[4*width*(idx+1) + 4*(jdx+1) + kdx];
	//	if(d_in[4*width*idx + 4*(jdx+1) + kdx]>max) max = d_in[4*width*idx + 4*(jdx+1) + kdx];
	//	d_out[new_width*idx + jdx*2 + kdx] = max;
	//	if(blockIdx.x == 0)printf("Not max = %d and stored %d at %d, at (%d,%d,%d) for index = %d\n",max,d_out[new_width*idx + jdx*2 + kdx],new_width*idx + jdx*2 + kdx,idx,jdx,kdx,index);
	//}
	//if(jdx % 2 == 0 && kdx == 3){
	//	d_out[new_width * idx + jdx*2 + 3] = d_in[4*width*idx + 4*jdx + 3];
	//}
	if(index < width/4){
		d_out[index] = index;
		if(index<100)printf("This is the index %d and this is d_out\n",index,d_out[index]);
	}
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

	const int block_quantity = (size+(BLOCK_WIDTH-1))/(BLOCK_WIDTH * 2 * 4);
	new_image = (unsigned char *)malloc(new_size);


	// declare GPU memory pointers
	unsigned char * d_in;
	unsigned char * d_out;

	// allocate GPU memory
	cudaMalloc(&d_in, size);
	cudaMalloc(&d_out, new_size);

	// transfer the array to the GPU
	cudaMemcpy(d_in, image, size, cudaMemcpyHostToDevice);

	printf("%d total threads in %d blocks of size %d\n",size, block_quantity, BLOCK_WIDTH);

	// launch the kernel
	dim3 dimBlock(BLOCK_WIDTH, 2, 4);
	dim3 dimGrid(block_quantity, 1, 1);

	pool<<<dimGrid, dimBlock>>>(d_out, d_in);

	// copy back the result array to the CPU
	cudaMemcpy(new_image, d_out, new_size, cudaMemcpyDeviceToHost);

	cudaFree(d_in);
	cudaFree(d_out);

	lodepng_encode32_file(output_filename, new_image, width, height);
	int i;
	for(i = 0; i<1000;i++)printf("new_image[%d] = %d\n",i,new_image[i]);

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
