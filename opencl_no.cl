// OpenCL kernel. Each work item takes care of one element of c
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void mat_comp( __global float *a, __global float *b, __global float *c, const unsigned int n, const unsigned int iteration ) {
	int i = get_global_id(0);
	int j = get_global_id(1);
	int t;
	
	float sum = 0.0f;
	
	for( t = j+1; t < j+11 && t < n; t++ ){
		sum += sqrt(a[i*n+t]);
	}
	
	c[i*n+j] = b[i*n+j] * sum * iteration;

} 

// Sequential matrix computation
__kernel void seq_mat_comp( __global float *a, __global float *b, __global float *c, const unsigned int n, const unsigned int iteration ) {
	
  int i,j,t;
  float sum = 0.0f;
  
  for( i = 0; i < n; i++ ){
	  for( j = 0; j < n; j++ ){
		  sum = 0.0f;
		  for( t = j+1; t < j+11 && t < n; t++ ){
			  sum += sqrt(a[i*n+t]);
		  }
		  c[i*n+j] = b[i*n + j] * sum * iteration;
	  }
  }
}
