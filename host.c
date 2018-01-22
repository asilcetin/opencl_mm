#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <CL/opencl.h>

#define MATRIXOUTPUT 1
#define MATRIXSIZE 64
#define LOCALSIZE 8

#define MAX_SOURCE_SIZE (0x100000)

int main( int argc, char* argv[] )
{

    srand(time(NULL));

    // Timers
    struct timeval Tvalue;
    struct timezone dummy;

    // Length of matrix and rows/cols
    unsigned int n = MATRIXSIZE;
    unsigned int sqrtN = sqrt(n);

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

    cl_platform_id cpPlatform;        // OpenCL platform
    cl_device_id device_id;           // device ID
    cl_context context;               // context
    cl_command_queue queue;           // command queue
    cl_program program;               // program
    cl_kernel kernel;                 // kernel

    // Size, in bytes, of each vector
    size_t bytes = n*sizeof(float);

    // Allocate memory for each vector on host
    h_a = (float*)malloc(bytes);
    h_b = (float*)malloc(bytes);
    h_c = (float*)malloc(bytes);

    // Initialize vectors on host
    int i;
    for( i = 0; i < n; i++ )
    {
        h_a[i] = ((float)rand())/RAND_MAX;
        h_b[i] = ((float)rand())/RAND_MAX;
    }

    if (MATRIXOUTPUT == 1) {
      // Print result
      for ( i = 0 ; i < n ; i++ ) {
          printf("%.2f ", h_a[i]);
          if ((i+1)%sqrtN == 0) { printf("\n"); }
      }
      printf("\n");
      for ( i = 0 ; i < n ; i++ ) {
          printf("%.2f ", h_b[i]);
          if ((i+1)%sqrtN == 0) { printf("\n"); }
      }
    }

    size_t globalSize, localSize;
    cl_int err;

    // Number of work items in each local work group
    localSize = LOCALSIZE;

    // Number of total work items - localSize must be devisor
    globalSize = ceil(n/(float)localSize)*localSize;

    printf(">>> Global size: %d\n", globalSize);
    printf(">>> Local size: %d\n", localSize);

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
    err = clGetPlatformIDs(1, &cpPlatform, NULL);

    // Get ID for the device
    err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

    // Create a context  
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);

    // Create a command queue 
    queue = clCreateCommandQueue(context, device_id, 0, &err);

    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1,
                            (const char **) & source_str, NULL, &err);

    // Build the program executable 
    clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

    // Create the compute kernel in the program we wish to run
    kernel = clCreateKernel(program, "mat_mul", &err);

    // Create the input and output arrays in device memory for our calculation
    d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);

    // Write our data set into the input array in device memory
    err = clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0,
                                   bytes, h_a, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0,
                                   bytes, h_b, 0, NULL, NULL);

    // Start the timing
    printf(">>> Starting calculation\n");
    gettimeofday(&Tvalue, &dummy);
    double starttime = (double)Tvalue.tv_sec + 1.0e-6*((double)Tvalue.tv_usec);

    // Set the arguments to our compute kernel
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_c);
    err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &n);

    // Execute the kernel over the entire range of the data set  
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize,
                                                              0, NULL, NULL);

    // Wait for the command queue to get serviced before reading back results
    clFinish(queue);

    // Read the results from the device
    clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0,
                                bytes, h_c, 0, NULL, NULL );

    if (MATRIXOUTPUT == 1) {
      // Print result
      for ( i = 0 ; i < n ; i++ ) {
          printf("%.2f ", h_c[i]);
          if ((i+1)%sqrtN == 0) { printf("\n"); }
      }
    }

    // End the timed loop
    gettimeofday(&Tvalue, &dummy);
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

    return 0;
}
