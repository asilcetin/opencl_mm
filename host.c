#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <CL/opencl.h>

#define MATRIX_OUTPUT 0
#define DEBUG_RESULT_OUTPUT 1
#define DEBUG_I 5834
#define DEBUG_J 2499
#define MATRIX_DIM_N 8192
#define LOCAL_SIZE 32
#define GLOBAL_SIZE 8192
#define ITERATION 20
#define SELECTED_PLATFORM_INDEX 0
#define SELECTED_DEVICE_INDEX 0

#define MAX_SOURCE_SIZE (0x100000)

int main( int argc, char* argv[] )
{
	
    srand(time(NULL));

    // Timers
    struct timeval Tvalue;
    struct timezone dummy;

    // Length of matrix and rows/cols
    unsigned int n = MATRIX_DIM_N;
    unsigned int matrix_length = n*n;

    // Host input vectors
    float *h_a;
    float *h_b;
    // Host output vector
    float *h_c;

    // Device input buffers
    cl_mem d_a;
    cl_mem d_b;
    // Device output buffer
    cl_mem d_c;

    cl_device_id device_id;           // device ID
	cl_platform_id *platforms;
	cl_uint num_platforms;
	cl_device_id *device_list;
	cl_uint num_devices;
    cl_context context;               // context
    cl_command_queue queue;           // command queue
    cl_program program;               // program
    cl_kernel kernel;                 // kernel

    // Size, in bytes, of each vector
    size_t bytes = matrix_length*sizeof(float);

    // Allocate memory for each vector on host
    h_a = (float*)malloc(bytes);
    h_b = (float*)malloc(bytes);
    h_c = (float*)malloc(bytes);

    // Initialize vectors on host
    int i;
	int j;
    for( i = 0; i < matrix_length; i++ )
    {
        h_a[i] = ((float)rand())/RAND_MAX;
        h_b[i] = ((float)rand())/RAND_MAX;
    }

    if (MATRIX_OUTPUT == 1) {
      // Print result
		for( i = 0 ; i < n ; i++ ) {
			for( j = 0 ; j < n ; j++ ) {
				printf("%.2f ", h_a[i*n+j]); 
			}
			printf("\n");
		}
		printf("\n");
		for( i = 0 ; i < n ; i++ ) {
			for( j = 0 ; j < n ; j++ ) {
				printf("%.2f ", h_b[i*n+j]); 
			}
			printf("\n");
		}
    }
	
	if(DEBUG_RESULT_OUTPUT){
		printf("Debug A[%d;%d] - A[%d;%d]:\n",DEBUG_I, DEBUG_J+1, DEBUG_I, DEBUG_I+10);
		i = DEBUG_I;
		for( j = DEBUG_J + 1 ; j < DEBUG_J + 11 ; j++ ) {
			printf("sqrt(%.4f)+", h_a[i*n+j]); 
		}
		printf("0\n\n");
	}
	
	if(DEBUG_RESULT_OUTPUT){
		printf("Debug B[%d;%d]:\n",DEBUG_I, DEBUG_J);
		i = DEBUG_I;
		j = DEBUG_J;
		printf("%.4f ", h_b[i*n+j]); 
		
		printf("\n\n");
	}

    cl_int err;

    // Number of work items in each local work group
	size_t local_size[2]  = {LOCAL_SIZE, LOCAL_SIZE};

    // Number of total work items - local size must be devisor
	size_t global_size[2] = {GLOBAL_SIZE, GLOBAL_SIZE};

    printf(">>> Global size: %d\n", GLOBAL_SIZE);
    printf(">>> Local size: %d\n", LOCAL_SIZE);

    // Load the kernel source code into the array source_str
    FILE *fp;
    char *source_str;
    size_t source_size;

    fp = fopen("opencl_no.cl", "r");
    if (!fp) {
        fprintf(stderr, ">>> Failed to load kernel\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose( fp );
    printf(">>> Kernel loading done\n");

    // Configure the OpenCL environment
    printf(">>> Initializing OpenCL\n");
	
    // Bind to platform
    err = clGetPlatformIDs(0, NULL, &num_platforms);
	platforms = (cl_platform_id *) malloc(sizeof(cl_platform_id) * num_platforms);
    err = clGetPlatformIDs(num_platforms, platforms, NULL);

    // Get ID for the device
	err = clGetDeviceIDs(platforms[SELECTED_PLATFORM_INDEX], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
	device_list = (cl_device_id *) malloc(sizeof(cl_device_id)*num_devices);
    err = clGetDeviceIDs(platforms[SELECTED_PLATFORM_INDEX], CL_DEVICE_TYPE_GPU, num_devices, device_list, NULL);

    // Create a context  
    context = clCreateContext(0, 1, device_list, NULL, NULL, &err);

    // Create a command queue 
    queue = clCreateCommandQueue(context, device_list[SELECTED_DEVICE_INDEX], 0, &err);

	printf(">>> Selected platform index: %d\n", SELECTED_PLATFORM_INDEX);
	printf(">>> Selected device index: %d\n", SELECTED_DEVICE_INDEX);
	printf(">>> Number of iterations: %d\n", ITERATION);
	
    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1,
                            (const char **) & source_str, NULL, &err);

    // Build the program executable 
    clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

    // Create the compute kernel in the program we wish to run
    kernel = clCreateKernel(program, "mat_comp", &err);

	if(err != CL_SUCCESS){
		printf("Error at clCreateKernel(), error code: %d\n", err);
		return 1;
	}
	
    // Create the input and output arrays in device memory for our calculation
    d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);

    // Write our data set into the input array in device memory
    err = clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0,
                                   bytes, h_a, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0,
                                   bytes, h_b, 0, NULL, NULL);

	printf(">>> Starting calculation\n");
    // Start the timing
	gettimeofday(&Tvalue, &dummy);
    double starttime = (double)Tvalue.tv_sec + 1.0e-6*((double)Tvalue.tv_usec);

	unsigned int iter = ITERATION;
	
    // Set the arguments to our compute kernel
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_c);
    err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &n);
    err |= clSetKernelArg(kernel, 4, sizeof(unsigned int), &iter);

    // Execute the kernel over the entire range of the data set  
	
	err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size, local_size,
                                                              0, NULL, NULL);
	
    
	if(err != CL_SUCCESS){
		printf("Error at clEnqueueNDRangeKernel(), error code: %d\n", err);
		return 1;
	}

    // Wait for the command queue to get serviced before reading back results
    clFinish(queue);

    // Read the results from the device
    clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0,
                                bytes, h_c, 0, NULL, NULL );

    // End the timed loop
    gettimeofday(&Tvalue, &dummy);
	
	if (MATRIX_OUTPUT == 1) {
      // Print result
      for( i = 0 ; i < n ; i++ ) {
		for( j = 0 ; j < n ; j++ ) {
			printf("%.2f ", h_c[i*n+j]); 
		}
		printf("\n");
		}
    }
	
	if(DEBUG_RESULT_OUTPUT){
		printf("Debug C[%d;%d]:\n",DEBUG_I, DEBUG_J);
		i = DEBUG_I;
		j = DEBUG_J;
		printf("%.2f ", h_c[i*n+j]); 
		
		printf("\n\n");
	}
	
    double endtime = (double)Tvalue.tv_sec + 1.0e-6*((double)Tvalue.tv_usec);
    double runtime = (endtime - starttime);
    printf(">>> Done: took %.5lf seconds runtime\n", runtime);

    // release OpenCL resources
    clReleaseMemObject(d_a);
    clReleaseMemObject(d_b);
    clReleaseMemObject(d_c);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    //release host memory
    free(h_a);
    free(h_b);
    free(h_c);
	
	free(platforms);

    return 0;
}