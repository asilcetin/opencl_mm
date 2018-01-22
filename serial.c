#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define MATRIXOUTPUT 1
#define nSIZE 64

// Matrix multiplication function
void mat_mul(float *a, float *b, float *c, int n) {
  int id, k;
  //printf("n: %d\n", n);

  for (id=0; id<n; id++){
    //Get our col and row
    int sqrtN = sqrt(n);
    int row = floor(id/sqrtN);
    int col = id%sqrtN;

    //printf("id: %d\n", id);
    //printf("row: %d\n", row);
    //printf("col: %d\n", col);

    for (k=0; k<sqrtN; k++){
      c[id] += a[row*sqrtN+k] * b[k*sqrtN+col];
      //printf("c_id: %d\n", id);
      //printf("c_val: %.2f\n", c[id]);
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

  // Length of vectors
  int n = nSIZE;
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
  printf(">>> Starting calculation\n");
  gettimeofday(&Tvalue, &dummy);
  float starttime = (float)Tvalue.tv_sec + 1.0e-6*((float)Tvalue.tv_usec);

  // Calculation
  mat_mul(h_a, h_b, h_c, n);

  // End the timed loop
  gettimeofday(&Tvalue, &dummy);
  float endtime = (float)Tvalue.tv_sec + 1.0e-6*((float)Tvalue.tv_usec);
  float runtime = (endtime - starttime);
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