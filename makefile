CC = nvcc
CFLAGS  = -I. -lm

# typing 'make' will invoke the first target entry in the file 
# (in this case the default target entry)
# you can name this target entry anything, but "default" or "all"
# are the most commonly used names by convention
#
default: all

all: test_equality rectify rectify_sequential pool pool_sequential convolve  convolve_sequential grid grid_sequential

nogrid: test_equality rectify rectify_sequential pool pool_sequential convolve convolve_sequential

# To create the executable file count we need the object files
#

rectify:  rectify.o lodepng.o
	$(CC) -o rectify rectify.o lodepng.o $(CFLAGS)

rectify.o: rectify.cu
	$(CC) -c rectify.cu

rectify_sequential:  rectify_sequential.o lodepng.o
	$(CC) -o rectify_sequential rectify_sequential.o lodepng.o $(CFLAGS)

rectify_sequential.o: rectify_sequential.cu
	$(CC) -c rectify_sequential.cu

pool:  pool.o lodepng.o
	$(CC) -o pool pool.o lodepng.o $(CFLAGS)

pool.o: pool.cu
	$(CC) -c pool.cu

pool_sequential:  pool_sequential.o lodepng.o
	$(CC) -o pool_sequential pool_sequential.o lodepng.o $(CFLAGS)

pool_sequential.o: pool_sequential.cu
	$(CC) -c pool_sequential.cu

convolve:  convolve.o lodepng.o
	$(CC) -o convolve convolve.o lodepng.o $(CFLAGS)

convolve.o: convolve.cu
	$(CC) -c convolve.cu

convolve_sequential:  convolve_sequential.o lodepng.o
	$(CC) -o convolve_sequential convolve_sequential.o lodepng.o $(CFLAGS)

convolve_sequential.o: convolve_sequential.cu
	$(CC) -c convolve_sequential.cu

test_equality:  test_equality.o lodepng.o
	$(CC) -o test_equality test_equality.o lodepng.o $(CFLAGS)

test_equality.o: test_equality.cu
	$(CC) -c test_equality.cu

lodepng.o: lodepng.cu
	$(CC)  -c lodepng.cu

grid:  grid.o
	$(CC) -o grid grid.o $(CFLAGS)

grid.o: grid.cu
	$(CC) -c grid.cu

grid_sequential:  grid_sequential.o
	$(CC) -o grid_sequential grid_sequential.o $(CFLAGS)

grid_sequential.o: grid_sequential.cu
	$(CC) -c grid_sequential.cu


# To start over from scratch, type 'make clean'.  This
# removes the executable file, as well as old .o object
# files and *~ backup files:
#
clean: 
	$(RM) count *.o *~
