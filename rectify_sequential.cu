/* Example of using lodepng to load, process, save image */
#include "lodepng.h"
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_WIDTH 1000
#define MAX_MSE 0.00001f


__device__ int counter; // initialise before running kernel


__global__ void rectify(unsigned char * d_out, unsigned char * d_in){
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	unsigned char f = d_in[idx];
	if(idx <3968050 && idx>3968030 ){
		printf("thread %d in block %d: idx = %d and f is %d\n", threadIdx.x, blockIdx.x, idx,f);
	}
	if(idx % 4 != 3){
		if (f < 127){
			f = 127;
			int cnt = atomicAdd(&counter, 1);
			printf("There was %d bytes changed to 127\n",cnt);
		}
	}
	d_out[idx] = f;
	if(idx <3968050 && idx>3968030 ){
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
	for(i = 3968030; i<3968050;i++){
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
	for(j = 3968030; j<3968050;j++){
		printf("This was image at %d: %d and now it is: %d and it is the %d value\n",j,image[j],new_image[j],j%4);
	}
	lodepng_encode32_file(output_filename, new_image, width, height);

	free(image);
	free(new_image);
	return 0;
}

float get_MSE(char* input_filename_1, char* input_filename_2)
{
  unsigned error1, error2;
  unsigned char *image1, *image2;
  unsigned width1, height1, width2, height2;

  error1 = lodepng_decode32_file(&image1, &width1, &height1, input_filename_1);
  error2 = lodepng_decode32_file(&image2, &width2, &height2, input_filename_2);
  if(error1) printf("error %u: %s\n", error1, lodepng_error_text(error1));
  if(error2) printf("error %u: %s\n", error2, lodepng_error_text(error2));
  if(width1 != width2) printf("images do not have same width\n");
  if(height1 != height2) printf("images do not have same height\n");

  // process image
  float im1, im2, diff, sum, MSE;
  sum = 0;
  int i;
  for (i = 0; i < width1 * height1 * 4; i++) {
    im1 = (float)image1[i];
    im2 = (float)image2[i];
    if (image1[i] - image2[i] != 0){
      printf("These are the two values: %d - %d at %d / %d\n",image1[i],image2[i],i,i%4);
    }
    diff = im1 - im2;
    sum += diff * diff;
  }
  int j;
  for(j = 3968030; j<3968050;j++){
    printf("This was image at %d: %d and now it is: %d and it is the %d value\n",j,image1[j],image2[j],j%4);
  }
  MSE = sqrt(sum) / (width1 * height1);

  free(image1);
  free(image2);

  return MSE;
}

int main(int argc, char *argv[])
{
	if ( argc >= 4 ){
		char* input_filename = argv[1];
		char* output_filename = argv[2];
		char* input_filename_test = argv[3];

		int error = process(input_filename, output_filename);

		if(error != 0){
			printf("An error occured. ( %d )\n",error);

		}else{
			printf("The rectification ran with success.\n");
			// get mean squared error between image1 and image2
			float MSE = get_MSE(input_filename, input_filename_test);

			if (MSE < MAX_MSE) {
				printf("Images are equal (MSE = %f, MAX_MSE = %f)\n",MSE,MAX_MSE);
			} else {
				printf("Images are NOT equal (MSE = %f, MAX_MSE = %f)\n",MSE,MAX_MSE);
			}
		}
	}else{
		printf("There is inputs missing.\n");
	}
	return 0;
}
