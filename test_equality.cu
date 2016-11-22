/* Check whether two images are the same */
#include "lodepng.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define MAX_MSE 0.00001f

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
  char* input_filename_1 = argv[1];
  char* input_filename_2 = argv[2];

  // get mean squared error between image1 and image2
  float MSE = get_MSE(input_filename_1, input_filename_2);

  if (MSE < MAX_MSE) {
    printf("Images are equal (MSE = %f, MAX_MSE = %f)\n",MSE,MAX_MSE);
  } else {
    printf("Images are NOT equal (MSE = %f, MAX_MSE = %f)\n",MSE,MAX_MSE);
  }

  return 0;
}
