// Code adapted from MATLAB implementation at https://people.ece.cornell.edu/land/courses/ece5760/LABS/s2016/lab3.html
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#define N 512 // grid side length
#define RHO 0.5 // related to pitch
#define ETA 2e-4 // related to duration of sound
#define BOUNDARY_GAIN 0.75 // clamped edge vs free edge

void print_grid(float **grid) {
	int i,j;
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			printf("(%d,%d): %f ", i,j,grid[i][j]);
		}
		printf("\n");
	}
	printf("\n");
}

int main(int argc, char** argv) {
	// get number of iterations to perform
	int T = atoi(argv[1]);
	struct timeval start_time, end_time;

	// initialize grid
	float **u = (float **) malloc(N * sizeof(float *));
	float **u1 = (float **) malloc(N * sizeof(float *));
	float **u2 = (float **) malloc(N * sizeof(float *));
	int i,j;
	for (i = 0; i < N; i++) {
		u[i] = (float *)malloc(N * sizeof(float)); 
		u1[i] = (float *)malloc(N * sizeof(float));
		u2[i] = (float *)malloc(N * sizeof(float));
		for (j = 0; j < N; j++) {
			u[i][j] = 0;
			u1[i][j] = 0;
			u2[i][j] = 0;
		}
	}
	printf("Size of grid: %d nodes\n", N*N);
	gettimeofday(&start_time, NULL);

	// simulate drum strike
	u1[N/2][N/2] = 1;
	float *audio = (float *)malloc(T * sizeof(float));
	float sum_of_neighbors, previous_value, previous_previous_value;
	float **temp;
	int t;
	for (t = 0; t < T; t++) {

		// update interior points
		int i,j;
		for (i = 1; i < N-1; i++) {
			for (j = 1; j < N-1; j++) {
				sum_of_neighbors = u1[i-1][j] + u1[i+1][j] + u1[i][j-1] + u1[i][j+1];
				previous_value = u1[i][j];
				previous_previous_value = u2[i][j];
				u[i][j] = (RHO * (sum_of_neighbors -4*previous_value) + 2*previous_value -(1-ETA)*previous_previous_value)/(1+ETA);
			}
		}

		// update side points
		for (i = 1; i < N-1; i++) {
			u[0][i] = BOUNDARY_GAIN * u[1][i]; // top
			u[N-1][i] = BOUNDARY_GAIN * u[N-2][i]; // bottom
			u[i][0] = BOUNDARY_GAIN * u[i][1]; // left
			u[i][N-1] = BOUNDARY_GAIN * u[i][N-2]; // right
		}

		// update corners
		u[0][0] = BOUNDARY_GAIN * u[1][0];
	    	u[N-1][0] = BOUNDARY_GAIN * u[N-2][0];
    		u[0][N-1] = BOUNDARY_GAIN * u[0][N-2];
    		u[N-1][N-1] = BOUNDARY_GAIN * u[N-1][N-2];

		

		// shift u1 into u2, u into u1
		// this is expensive!
		/*
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				u2[i][j] = u1[i][j];
				u1[i][j] = u[i][j];
			}
		}
		*/

//		print_grid(u);

		audio[t] = u[N/2][N/2];
		printf("%f,\n", audio[t]);

		temp = u2;
		u2 = u1;
		u1 = u;
		u = temp;

		// record displacement at node (N-1,N-1)
	}
	gettimeofday(&end_time, NULL);

	unsigned long long time_elapsed = 1000 * (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_usec - start_time.tv_usec) / 1000;

	printf("Time Elapsed [%llu ms]\n", time_elapsed);

	// free grid memory
	for (i = 0; i < N; i++) {
		free(u[i]);
		free(u1[i]);
		free(u2[i]);
	}

	free(u);
	free(u1);
	free(u2);
	free(audio);
}
