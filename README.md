# opencl_mm
OpenCl matrix calculation benchmarking

## Run
qlogin

gcc serial.c -o serial.out -lm

./serial.out

gcc -I/usr/local/cuda-5.0/include/ -lOpenCL host.c -o host.out -lm

./host.out
