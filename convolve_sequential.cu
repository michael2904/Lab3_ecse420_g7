/* Example of using lodepng to load, process, save image */
#include "lodepng.h"
#include <stdio.h>
#include <stdlib.h>
#include "wm.h"

void process(char* input_filename, char* output_filename)
{
  unsigned error;
  unsigned char *image, *new_image;
  unsigned width, height;

  error = lodepng_decode32_file(&image, &width, &height, input_filename);
  if(error) printf("error %u: %s\n", error, lodepng_error_text(error));
  new_image = malloc(width * height * 4 * sizeof(unsigned char));

  // convolve image
  int i,j,k,ii,jj;
  float value;
  for (i = 1; i < height-1; i++) {
    for (j = 1; j < width-1; j++) {	
      for (k = 0; k < 3; k++) {
        value = 0;
        for (ii = 0; ii < 3; ii++) {
          for (jj = 0; jj < 3; jj++) {
            value += image[4*width*(i+ii-1) + 4*(j+jj-1) + k] * w[ii][jj];
          }
        }
        value = value > 255 ? 255 : value;
        value = value < 0 ? 0 : value;
        new_image[4*(width-2)*(i-1) + 4*(j-1) + k] = value;
      }
      new_image[4*(width-2)*(i-1) + 4*(j-1) + 3] = image[4*width*i + 4*j + 3]; // A
    }
  }

  lodepng_encode32_file(output_filename, new_image, width-2, height-2);

  free(image);
  free(new_image);
}

int main(int argc, char *argv[])
{
  char* input_filename = argv[1];
  char* output_filename = argv[2];

  process(input_filename, output_filename);

  return 0;
}
