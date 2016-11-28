/* Example of using lodepng to load, process, save image */
#include "lodepng.h"
#include <stdio.h>
#include <stdlib.h>

void process(char* input_filename, char* output_filename)
{
  unsigned error;
  unsigned char *image, *new_image;
  unsigned width, height;

  error = lodepng_decode32_file(&image, &width, &height, input_filename);
  if(error) printf("error %u: %s\n", error, lodepng_error_text(error));
  new_image =(unsigned char *) malloc(width * height * 4 * sizeof(unsigned char));

  // rectify image
  int i,j,k;
  for (i = 0; i < height; i++) {
    for (j = 0; j < width; j++) {
      for (k = 0; k < 1000; k++) {
  	    new_image[4*width*i + 4*j + 0] = image[4*width*i + 4*j + 0] < 127 ? 127 : image[4*width*i + 4*j + 0]; // R
  	    new_image[4*width*i + 4*j + 1] = image[4*width*i + 4*j + 1] < 127 ? 127 : image[4*width*i + 4*j + 1]; // G
  	    new_image[4*width*i + 4*j + 2] = image[4*width*i + 4*j + 2] < 127 ? 127 : image[4*width*i + 4*j + 2]; // B
  	    new_image[4*width*i + 4*j + 3] = image[4*width*i + 4*j + 3]; // A
      }
    }
  }

  lodepng_encode32_file(output_filename, new_image, width, height);

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
