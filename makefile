CC = nvcc
CFLAGS  = -I. -lm

# typing 'make' will invoke the first target entry in the file 
# (in this case the default target entry)
# you can name this target entry anything, but "default" or "all"
# are the most commonly used names by convention
#
default: all

all: test_equality rectify_sequential

# To create the executable file count we need the object files
#

rectify_sequential:  rectify_sequential.o lodepng.o
	$(CC) -o rectify rectify_sequential.o lodepng.o $(CFLAGS)

rectify_sequential.o: rectify_sequential.cu
	$(CC) -c rectify_sequential.cu

test_equality:  test_equality.o lodepng.o
	$(CC) -o test_equality test_equality.o lodepng.o $(CFLAGS)

test_equality.o: test_equality.cu
	$(CC) -c test_equality.cu

lodepng.o: lodepng.cu
	$(CC)  -c lodepng.cu


# To start over from scratch, type 'make clean'.  This
# removes the executable file, as well as old .o object
# files and *~ backup files:
#
clean: 
	$(RM) count *.o *~
