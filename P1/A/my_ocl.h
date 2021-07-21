#ifndef _OCL_H

#define _OCL_H

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <unistd.h>
#else
#include <CL/cl.h>
#endif
#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_GPU
#endif

int remove_noiseOCL(float *im, float *image_out, 
	float thredshold, int window_size,
	int height, int width);
#endif
