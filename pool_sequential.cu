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
  new_image = (unsigned char *)malloc(width * height * sizeof(unsigned char) * 2);

  // pool image
  unsigned char max_val;
  int i,j,k;
  for (i = 0; i < height; i+=2) {
    for (j = 0; j < width; j+=2) {
      for (k = 0; k < 3; k++) {
        // k=0 -> R, k=1 -> G, k=2 -> B
        max_val = 0;
        if (image[4*width*i + 4*j + k] > max_val) max_val = image[4*width*i + 4*j + k];
        if (image[4*width*i + 4*(j+1) + k] > max_val) max_val = image[4*width*i + 4*(j+1) + k];
        if (image[4*width*(i+1) + 4*j + k] > max_val) max_val = image[4*width*(i+1) + 4*j + k];
        if (image[4*width*(i+1) + 4*(j+1) + k] > max_val) max_val = image[4*width*(i+1) + 4*(j+1) + k];
  	    new_image[width*i + 2*j + k] = max_val;
      }

	    new_image[width*i + 2*j + 3] = image[4*width*i + 4*j + 3]; // A
    }
  }
  
  lodepng_encode32_file(output_filename, new_image, width/2, height/2);

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
