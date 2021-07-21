#include <stdio.h>
#include "my_ocl.h"
#include "common.c"

int remove_noiseOCL(float *im, float *image_out, 
	float thredshold, int window_size,
	int height, int width)
{
    int          err;               // error code returned from OpenCL calls
    float        *h_a;              // a vector 
    float        *h_b;              // b vector 
    float        *h_c;              // c vector (a+b) returned from the compute device
    unsigned int correct;           // number of correct results  

    size_t global[2];                  // global domain size  

    cl_device_id     device_id;     // compute device id 
    cl_context       context;       // compute context
    cl_command_queue commands;      // compute command queue
    cl_program       program;       // compute program
    cl_kernel        ko_salt_pepper;// compute kernel
    
	cl_mem im_buffer;       
	cl_mem imageout_buffer;

	FILE *fp;
	long filelen;
	long readlen;
	char *kernel_src;  // char string to hold kernel source


	fp = fopen("pimienta.cl","r");
	fseek(fp,0L, SEEK_END);
	filelen = ftell(fp);
	rewind(fp);

	kernel_src = malloc(sizeof(char)*(filelen+1));
	readlen = fread(kernel_src,1,filelen,fp);
	if(readlen!= filelen)
	{
		printf("error reading file\n");
		exit(1);
	}

	// ensure the string is NULL terminated
	kernel_src[filelen+1]='\0';

	// Set up platform and GPU device

	cl_uint numPlatforms;

    // Find number of platforms
    err = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (err != CL_SUCCESS || numPlatforms <= 0)
    {
        printf("Error: Failed to find a platform!\n%s\n",err_code(err));
        return EXIT_FAILURE;
    }

    // Get all platforms
    cl_platform_id Platform[numPlatforms];
    err = clGetPlatformIDs(numPlatforms, Platform, NULL);
    if (err != CL_SUCCESS || numPlatforms <= 0)
    {
        printf("Error: Failed to get the platform!\n%s\n",err_code(err));
        return EXIT_FAILURE;
    }

    // Secure a GPU
    for (int i = 0; i < numPlatforms; i++)
    {
        err = clGetDeviceIDs(Platform[i], DEVICE, 1, &device_id, NULL);
        if (err == CL_SUCCESS)
        {
            break;
        }
    }

	 if (device_id == NULL)
    {
        printf("Error: Failed to create a device group!\n%s\n",err_code(err));
        return EXIT_FAILURE;
    }

    err = output_device_info(device_id);
  
    // Create a compute context 
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context)
    {
        printf("Error: Failed to create a compute context!\n%s\n", err_code(err));
        return EXIT_FAILURE;
    }

    // Create a command queue
    commands = clCreateCommandQueue(context, device_id, 0, &err);
    if (!commands)
    {
        printf("Error: Failed to create a command commands!\n%s\n", err_code(err));
        return EXIT_FAILURE;
    }

    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1, (const char **) & kernel_src, NULL, &err);
    if (!program)
    {
        printf("Error: Failed to create compute program!\n%s\n", err_code(err));
        return EXIT_FAILURE;
    }

    // Build the program  
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n%s\n", err_code(err));
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        return EXIT_FAILURE;
    }

    // Create the compute kernel from the program 
    ko_salt_pepper = clCreateKernel(program, "remove_noise", &err);
    if (!ko_salt_pepper || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel!\n%s\n", err_code(err));
        return EXIT_FAILURE;
    }


	// create buffer objects to input and output args of kernel function
	im_buffer       = clCreateBuffer(context, CL_MEM_READ_ONLY, width*height*sizeof(float),  NULL, NULL);
	imageout_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, width*height*sizeof(float),  NULL, NULL);
	
	err = clEnqueueWriteBuffer(commands, im_buffer, CL_TRUE, 0, width*height*sizeof(float), im, 0, NULL, NULL );  
	if (err != CL_SUCCESS)
	{	
		// printf("Error enqueuing read buffer command. Error Code=%s\n",err_code(err));
		exit(1);
	}

	// set the kernel arguments
	if ( clSetKernelArg(ko_salt_pepper, 0, sizeof(cl_mem), &im_buffer) ||
			clSetKernelArg(ko_salt_pepper, 1, sizeof(cl_mem), &imageout_buffer)  || 
			clSetKernelArg(ko_salt_pepper, 2, sizeof(cl_float), &thredshold)  ||
			clSetKernelArg(ko_salt_pepper, 3, sizeof(cl_int), &window_size)  ||
			clSetKernelArg(ko_salt_pepper, 4, sizeof(cl_int), &height)  ||
			clSetKernelArg(ko_salt_pepper, 5, sizeof(cl_int), &width)  != CL_SUCCESS)
	{
		printf("Unable to set kernel arguments. Error Code=%d\n",err);
		exit(1);
	}

	// set the global work dimension size
	global[0]= width;
	global[1]= height;

	// Enqueue the kernel object with 
	// Dimension size = 2, 
	// global worksize = global, 
	// local worksize = NULL - let OpenCL runtime determine
	// No event wait list
	//printf("Enviado kernel al device\n");
	
	// double t0d = getMicroSeconds();
	err = clEnqueueNDRangeKernel(commands, ko_salt_pepper, 2, NULL, 
							global, NULL, 0, NULL, NULL);
	// double t1d = getMicroSeconds();

	if (err != CL_SUCCESS)
	{	
		printf("Unable to enqueue kernel command. Error Code=%d\n",err);
		exit(1);
	}

	err = clEnqueueReadBuffer(commands, imageout_buffer, CL_TRUE, 0, width*height*sizeof(float), image_out, 0, NULL, NULL );  
	if (err != CL_SUCCESS)
	{	
		printf("Error enqueuing read buffer command.");
		exit(1);
	}

	// clean up
	clReleaseProgram(program);
	clReleaseKernel(ko_salt_pepper);
	clReleaseCommandQueue(commands);
	clReleaseContext(context);
	free(kernel_src);


	// return 0;


/*****/
	
}

