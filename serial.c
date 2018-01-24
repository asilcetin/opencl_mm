#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define MATRIXOUTPUT 1
#define MATRIXSIZE 64
#define ITERATION 1


// Matrix computation
void mat_comp(float *a, float *b, float *c, int n) {
	
  int i,j,t;
  float sum;
  
  for( i = 0; i < n; i++ ){
	  for( j = 0; j < n; j++ ){
		  sum = 0.0f;
		  for( t = j+1; t < j+11 && t < n; t++ ){
			  sum += sqrt(a[i*n+t]);
		  }
		  c[i*n+j] += b[i*n + j] * sum;
	  }
  }
}

int main()
{

  srand(time(NULL));

  // Timers
  struct timeval Tvalue;
  struct timezone dummy;

  // Variables
  int i, j, k;

  // Length of matrix and rows/cols
  int n = MATRIXSIZE;
  int sqrtN = sqrt(n);

  // Host input vectors
  float *h_a;
  float *h_b;
  // Host output vector
  float *h_c;

  // Size, in bytes, of each vector
  size_t bytes = n*sizeof(float);

  // Allocate memory for each vector on host
  h_a = (float*)malloc(bytes);
  h_b = (float*)malloc(bytes);
  h_c = (float*)malloc(bytes);

  // Initialize vectors on host
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

  // Start the timing
  printf(">>> Starting calculation for %d iteration(s)\n", ITERATION);
  gettimeofday(&Tvalue, &dummy);
  double starttime = (double)Tvalue.tv_sec + 1.0e-6*((double)Tvalue.tv_usec);

  for( i = 0; i < ITERATION; i++ ){
	// Calculation
	mat_comp(h_a, h_b, h_c, sqrtN);
  }
  
  // End the timed loop
  gettimeofday(&Tvalue, &dummy);
  double endtime = (double)Tvalue.tv_sec + 1.0e-6*((double)Tvalue.tv_usec);
  double runtime = (endtime - starttime);
  printf(">>> Done: took %.5lf seconds runtime\n", runtime);

  if (MATRIXOUTPUT == 1) {
    // Print the result
    for ( i = 0 ; i < n ; i++ ) {
        printf("%.2f ", h_c[i]);
        if ((i+1)%sqrtN == 0) { printf("\n"); }
    }
  }

  return 0;
}
