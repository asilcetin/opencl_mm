// OpenCL kernel. Each work item takes care of one element of c
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void mat_comp(  __global float *a, __global float *b, __global float *c, const unsigned int n) {
	int i = get_global_id(0);
	int j = get_global_id(1);
	int t;
	
	float sum;
	
	for( t = j+1; t < j+11 && t < n; t++ ){
		sum += sqrt(a[i*n+t]);
	}
	
	c[i*n+j] += b[i*n + j] * sum;

} 
