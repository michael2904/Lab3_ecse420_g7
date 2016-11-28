/* Example of using lodepng to load, process, save image */
#include "lodepng.h"
#include <stdio.h>
#include <stdlib.h>
#include "wm.h"
#define MAX_MSE 0.00001f


#define BLOCK_WIDTH 512

//Putting blocks of size width divided by 0, so that each thread can access the neighboring values. There is no neighboring value that is called twice.

__global__ void convolve(unsigned char * d_out, unsigned char * d_in,int width,int height,float w[3][3]){

	int ind = blockIdx.x * blockDim.x + threadIdx.x;
	int i = ((ind) / (width*4))+1;
	int j = ((ind/4) % (width))+1;
	int k = (ind) % 4;
	int ii,jj;
	if(ind == 0){
		for (ii = 0; ii < 3; ii++) {
			for (jj = 0; jj < 3; jj++) {
				printf("ind %d w(%d,%d)=%f|",ind,ii,jj,w[ii][jj]);
			}
			printf("\n");
		}
	}
	if(k != 3){
		float currentWF = 0;
		float value = 0;
		for (ii = 0; ii < 3; ii++) {
			for (jj = 0; jj < 3; jj++) {
				currentWF = w[ii][jj];
				value += d_in[4*width*(i+ii-1) + 4*(j+jj-1) + k] * currentWF;
				if(ind>3952120)printf("ind %d w(%d,%d)=%f|value : %f|\n",ind,ii,jj,w[ii][jj],value);
			}
			if(ind>3952120)printf("\n");
		}
		if(ind>3952120)printf("\nind %d value : %f\n",ind,value);
		value = value > 255 ? 255 : value;
		value = value < 0 ? 0 : value;
		if(ind>3952120)printf("\nind %d value : %f\n",ind,value);
		d_out[4*(width-2)*(i-1) + 4*(j-1) + k] = (unsigned char) value;
	}else{
		d_out[4*(width-2)*(i-1) + 4*(j-1) + 3] = d_in[4*width*i + 4*j + 3]; // A
	}
}


int process(char* input_filename, char* output_filename,char* test_filename){
	unsigned error,error2;
	unsigned char *image, *new_image, *test_image;
	unsigned width, height;
	unsigned new_width, new_height;
	unsigned t_width, t_height;

	error = lodepng_decode32_file(&image, &width, &height, input_filename);
	if(error){
		printf("error %u: %s\n", error, lodepng_error_text(error));
		return error;
	}
	error2 = lodepng_decode32_file(&test_image, &t_width, &t_height, test_filename);
	if(error2){
		printf("error %u: %s\n", error2, lodepng_error_text(error2));
		return error2;
	}
	new_width = (width)-2;
	new_height = (height)-2;

	const int size = width * height * 4 * sizeof(unsigned char);
	const int new_size = new_width * new_height * 4 * sizeof(unsigned char);

	new_image = (unsigned char *)malloc(new_size);


	// declare GPU memory pointers
	unsigned char * d_in;
	unsigned char * d_out;
	float *w_d = 0;

	// allocate GPU memory
	cudaMalloc(&d_in, size);
	cudaMalloc(&d_out, new_size);
	cudaMalloc((void**) &w_d, 9 * sizeof(float));

	// transfer the array to the GPU
	cudaMemcpy(d_in, image, size, cudaMemcpyHostToDevice);
	cudaMemcpy (w_d, w, 9 * sizeof(float), cudaMemcpyHostToDevice);


	printf("%d total size with width %d and height %d in %d blocks of size %d\n",size,width,height, (size+(BLOCK_WIDTH-1))/BLOCK_WIDTH, BLOCK_WIDTH);

	// launch the kernel
	dim3 dimGrid((new_size+(BLOCK_WIDTH-1))/BLOCK_WIDTH);
	dim3 dimBlock(BLOCK_WIDTH);

	int ii,jj;
	for (ii = 0; ii < 3; ii++) {
		for (jj = 0; jj < 3; jj++) {
			printf("w(%d,%d)=%f|",ii,jj,w[ii][jj]);
		}
		printf("\n");
	}
	convolve<<<dimGrid, dimBlock>>>(d_out, d_in,width,height,(float(*) [3])w_d);

	// copy back the result array to the CPU
	cudaMemcpy(new_image, d_out, new_size, cudaMemcpyDeviceToHost);

	cudaFree(d_in);
	cudaFree(d_out);

	if (cudaGetLastError() != cudaSuccess) printf("kernel launch failed\n");

	cudaThreadSynchronize();

	if (cudaGetLastError() != cudaSuccess) printf("kernel execution failed\n");

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
	if(new_width != t_width) printf("images do not have same width\n");
  	if(new_height != t_height) printf("images do not have same height\n");

  	// process image
	float im1, im2, diff, sum, MSE;
	sum = 0;
	for (i = 0; i < width1 * height1 * 4; i++) {
		im1 = (float)new_image[i];
		im2 = (float)test_image[i];
		if (new_image[i] - test_image[i] != 0){
			printf("These are the two values: %d - %d at %d / %d\n",new_image[i],test_image[i],i,i%4);
		}
		diff = im1 - im2;
		sum += diff * diff;
	}

	MSE = sqrt(sum) / (new_width * new_height);
	if (MSE < MAX_MSE) {
		printf("Images are equal (MSE = %f, MAX_MSE = %f)\n",MSE,MAX_MSE);
	} else {
		printf("Images are NOT equal (MSE = %f, MAX_MSE = %f)\n",MSE,MAX_MSE);
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
