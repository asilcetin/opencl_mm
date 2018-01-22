// OpenCL kernel. Each work item takes care of one element of c
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void mat_mul(  __global float *a, __global float *b, __global float *c, const unsigned int n) {

  int k;
  int id = get_global_id(0);

  //Get our col and row
  int sqrtN = sqrt((float)n);
  int row = floor( (float)id / (float)sqrtN );
  int col = id%sqrtN;

  float tmp = 0.00;
  for (k=0; k<sqrtN; k++){
    tmp += a[row*sqrtN+k] * b[k*sqrtN+col];
  }
  c[id] = tmp;

}
