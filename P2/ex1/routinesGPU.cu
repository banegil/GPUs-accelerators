#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#include "routinesGPU.h"

#define BLOCKSIZE 16

__global__ void calculateNR(uint8_t  *im, float *NR, int height, int width) {
    // Definition
    int i, j, bi, bj;

    // Retrieve global id
    i = threadIdx.y + blockDim.y*blockIdx.y + 2;
    j = threadIdx.x + blockDim.x*blockIdx.x + 2;  

    // Retrieve id within block
    bi = threadIdx.y + 2;
    bj = threadIdx.x + 2;

    /* Shared memory preparation */
    __shared__ float im_shared[2 + int(BLOCKSIZE) + 2][2 + int(BLOCKSIZE) + 2];

    // Load left superior corner
    if (2 <= bi && bi < 4 && 2 <= bj && bj < 4){
        im_shared[bi-2][bj-2] = im[(i-2)*width + j - 2];
    }

    // Load left superior corner
    if (2 <= bi && bi < 4 && 2 <= bj && bj < 4){
        im_shared[bi-2][bj-2] = im[(i-2)*width + j - 2];
    }

    // Load right superior corner
    if (2 <= bi && bi < 4 && BLOCKSIZE <= bj && bj < BLOCKSIZE + 2){
        im_shared[bi-2][bj+2] = im[(i-2)*width + j + 2];
    }

    // Load left inferior corner
    if (BLOCKSIZE <= bi && bi <= BLOCKSIZE + 2 && 2 <= bj && bj < 4){
        im_shared[bi+2][bj-2] = im[(i+2)*width + j - 2];
    }

    // Load right inferior corner
    if (BLOCKSIZE <= bi && bi <= BLOCKSIZE + 2 && BLOCKSIZE <= bj && bj < BLOCKSIZE + 2){
        im_shared[bi+2][bj+2] = im[(i+2)*width + j + 2];
    }

    // Load superior edge
    if (2 <= bi && bi < 4){
        im_shared[bi-2][bj] = im[(i-2)*width + j];
    }

    // Load inferior edge
    if (BLOCKSIZE <= bi && bi <= BLOCKSIZE + 2){
        im_shared[bi+2][bj] = im[(i+2)*width + j];
    }

    // Load left edge
    if (2 <= bj && bj < 4){
        im_shared[bi][bj-2] = im[i*width + j - 2];
    }

    // Load right edge
    if (BLOCKSIZE <= bj && bj < BLOCKSIZE + 2){
        im_shared[bi][bj+2] = im[i*width + j + 2];
    }

    // Load center data
    im_shared[bi][bj] = im[i*width + j];

    __syncthreads();

    // Noise reduction
    if (i < height-2 && j < width - 2) 
    {
        NR[i*width+j] =
             (2.0*im_shared[bi-2][bj-2] +  4.0*im_shared[bi-2][bj-1] +  5.0*im_shared[bi-2][bj] +  4.0*im_shared[bi-2][bj+1] + 2.0*im_shared[bi-2][bj+2]
            + 4.0*im_shared[bi-1][bj-2] +  9.0*im_shared[bi-1][bj-1] + 12.0*im_shared[bi-1][bj] +  9.0*im_shared[bi-1][bj+1] + 4.0*im_shared[bi-1][bj+2]
            + 5.0*im_shared[bi  ][bj-2] + 12.0*im_shared[bi  ][bj-1] + 15.0*im_shared[bi  ][bj] + 12.0*im_shared[bi  ][bj+1] + 5.0*im_shared[bi  ][bj+2]
            + 4.0*im_shared[bi+1][bj-2] +  9.0*im_shared[bi+1][bj-1] + 12.0*im_shared[bi+1][bj] +  9.0*im_shared[bi+1][bj+1] + 4.0*im_shared[bi+1][bj+2]
            + 2.0*im_shared[bi+2][bj-2] +  4.0*im_shared[bi+2][bj-1] +  5.0*im_shared[bi+2][bj] +  4.0*im_shared[bi+2][bj+1] + 2.0*im_shared[bi+2][bj+2])
            /159.0;
    }
}


__global__ void calculateGPhi(float *NR, float *G, float *phi, float *Gx, float *Gy, int height, int width) {

	int i, j;
	float PI = 3.141593;

	j = blockIdx.x * blockDim.x + threadIdx.x;
	i = blockIdx.y * blockDim.y + threadIdx.y;
	//G[i*width+j] = 0;
	phi[i*width+j] = 0;
	if(((i >=2) && (i < height-2)) && ((j >=2) && (j < width-2))) {
		// Intensity gradient of the image
			Gx[i*width+j] = 
				 (1.0*NR[(i-2)*width+(j-2)] +  2.0*NR[(i-2)*width+(j-1)] +  (-2.0)*NR[(i-2)*width+(j+1)] + (-1.0)*NR[(i-2)*width+(j+2)]
				+ 4.0*NR[(i-1)*width+(j-2)] +  8.0*NR[(i-1)*width+(j-1)] +  (-8.0)*NR[(i-1)*width+(j+1)] + (-4.0)*NR[(i-1)*width+(j+2)]
				+ 6.0*NR[(i  )*width+(j-2)] + 12.0*NR[(i  )*width+(j-1)] + (-12.0)*NR[(i  )*width+(j+1)] + (-6.0)*NR[(i  )*width+(j+2)]
				+ 4.0*NR[(i+1)*width+(j-2)] +  8.0*NR[(i+1)*width+(j-1)] +  (-8.0)*NR[(i+1)*width+(j+1)] + (-4.0)*NR[(i+1)*width+(j+2)]
				+ 1.0*NR[(i+2)*width+(j-2)] +  2.0*NR[(i+2)*width+(j-1)] +  (-2.0)*NR[(i+2)*width+(j+1)] + (-1.0)*NR[(i+2)*width+(j+2)]);


			Gy[i*width+j] = 
				 ((-1.0)*NR[(i-2)*width+(j-2)] + (-4.0)*NR[(i-2)*width+(j-1)] +  (-6.0)*NR[(i-2)*width+(j)] + (-4.0)*NR[(i-2)*width+(j+1)] + (-1.0)*NR[(i-2)*width+(j+2)]
				+ (-2.0)*NR[(i-1)*width+(j-2)] + (-8.0)*NR[(i-1)*width+(j-1)] + (-12.0)*NR[(i-1)*width+(j)] + (-8.0)*NR[(i-1)*width+(j+1)] + (-2.0)*NR[(i-1)*width+(j+2)]
				+    2.0*NR[(i+1)*width+(j-2)] +    8.0*NR[(i+1)*width+(j-1)] +    12.0*NR[(i+1)*width+(j)] +    8.0*NR[(i+1)*width+(j+1)] +    2.0*NR[(i+1)*width+(j+2)]
				+    1.0*NR[(i+2)*width+(j-2)] +    4.0*NR[(i+2)*width+(j-1)] +     6.0*NR[(i+2)*width+(j)] +    4.0*NR[(i+2)*width+(j+1)] +    1.0*NR[(i+2)*width+(j+2)]);

			G[i*width+j]   = sqrtf((Gx[i*width+j]*Gx[i*width+j])+(Gy[i*width+j]*Gy[i*width+j]));	//G = √Gx²+Gy²
			phi[i*width+j] = atan2f(fabs(Gy[i*width+j]),fabs(Gx[i*width+j]));

			if(fabs(phi[i*width+j])<=PI/8 )
				phi[i*width+j] = 0;
			else if (fabs(phi[i*width+j])<= 3*(PI/8))
				phi[i*width+j] = 45;
			else if (fabs(phi[i*width+j]) <= 5*(PI/8))
				phi[i*width+j] = 90;
			else if (fabs(phi[i*width+j]) <= 7*(PI/8))
				phi[i*width+j] = 135;
			else phi[i*width+j] = 0;
		
	}




}

__global__ void calculatePedge(float *G, float *phi, uint8_t *pedge, int height, int width) {
	
	int i, j;

	j = blockIdx.x * blockDim.x + threadIdx.x;
	i = blockIdx.y * blockDim.y + threadIdx.y;
	pedge[i*width+j] = 0;
	if(((i >=3) && (i < height-3)) && ((j >=3) && (j < width-3))) {
		
			if(phi[i*width+j] == 0){
				if(G[i*width+j]>G[i*width+j+1] && G[i*width+j]>G[i*width+j-1]) //edge is in N-S
					pedge[i*width+j] = 1;

			} else if(phi[i*width+j] == 45) {
				if(G[i*width+j]>G[(i+1)*width+j+1] && G[i*width+j]>G[(i-1)*width+j-1]) // edge is in NW-SE
					pedge[i*width+j] = 1;

			} else if(phi[i*width+j] == 90) {
				if(G[i*width+j]>G[(i+1)*width+j] && G[i*width+j]>G[(i-1)*width+j]) //edge is in E-W
					pedge[i*width+j] = 1;

			} else if(phi[i*width+j] == 135) {
				if(G[i*width+j]>G[(i+1)*width+j-1] && G[i*width+j]>G[(i-1)*width+j+1]) // edge is in NE-SW
					pedge[i*width+j] = 1;
			}
	}
}

__global__ void calculateImageOut(uint8_t *image_out, float *G, uint8_t *pedge, float level, int height, int width) {

	float lowthres, hithres;
	int i, j;
	int ii, jj;

	j = blockIdx.x * blockDim.x + threadIdx.x;
	i = blockIdx.y * blockDim.y + threadIdx.y;

	// Hysteresis Thresholding
	lowthres = level/2;
	hithres  = 2*(level);
	image_out[i*width+j] = 0;
	if(((i >=3) && (i < height-3)) && ((j >=3) && (j < width-3))) {
		if(G[i*width+j]>hithres && pedge[i*width+j])
			image_out[i*width+j] = 255;
		else if(pedge[i*width+j] && G[i*width+j]>=lowthres && G[i*width+j]<hithres)
			// check neighbours 3x3
			for (ii=-1;ii<=1; ii++)
				for (jj=-1;jj<=1; jj++)
					if (G[(i+ii)*width+j+jj]>hithres)
						image_out[i*width+j] = 255;
	}
}

void cannyGPU(uint8_t *im, uint8_t *image_out, 
	float *NR, float *G, float *phi, float *Gx, float *Gy, uint8_t *pedge,
	float level, int height, int width)
{

	int nThreads_previo = 16;	
	dim3 nThreads(nThreads_previo, nThreads_previo);
	int myblocks;
	if (height%16==0)
		myblocks=height/16;
	else 
		myblocks = height/16+1;

	int myblocks2;
	if (width%16==0)
		myblocks2=width/16;
	else 
		myblocks2 = width/16+1;

	dim3 nBlocks(myblocks2, myblocks);

	
	calculateNR<<<nBlocks,nThreads>>>(im, NR, height, width);
	cudaDeviceSynchronize();

	calculateGPhi<<<nBlocks,nThreads>>>(NR, G, phi, Gx, Gy, height, width);
	cudaDeviceSynchronize();

	calculatePedge<<<nBlocks,nThreads>>>(G, phi, pedge, height, width);
	cudaDeviceSynchronize();
	
	// Edge
	calculateImageOut<<<nBlocks,nThreads>>>(image_out, G, pedge, level, height, width);
}

void houghtransform(uint8_t *im, int width, int height, uint32_t *accumulators, int accu_width, int accu_height, 
	float *sin_table, float *cos_table)
{
	int i, j, theta;

	float hough_h = ((sqrt(2.0) * (float)(height>width?height:width)) / 2.0);

	for(i=0; i<accu_width*accu_height; i++)
		accumulators[i]=0;	

	float center_x = width/2.0; 
	float center_y = height/2.0;
	for(i=0;i<height;i++)  
	{  
		for(j=0;j<width;j++)  
		{  
			if( im[ (i*width) + j] > 250 ) // Pixel is edge  
			{  
				for(theta=0;theta<180;theta++)  
				{  
					float rho = ( ((float)j - center_x) * cos_table[theta]) + (((float)i - center_y) * sin_table[theta]);
					accumulators[ (int)((round(rho + hough_h) * 180.0)) + theta]++;

				} 
			} 
		} 
	}
}

void getlines(int threshold, uint32_t *accumulators, int accu_width, int accu_height, int width, int height, 
	float *sin_table, float *cos_table,
	int *x1_lines, int *y1_lines, int *x2_lines, int *y2_lines, int *lines)
{
	int rho, theta, ii, jj;
	uint32_t max;

	for(rho=0;rho<accu_height;rho++)
	{
		for(theta=0;theta<accu_width;theta++)  
		{  

			if(accumulators[(rho*accu_width) + theta] >= threshold)  
			{  
				//Is this point a local maxima (9x9)  
				max = accumulators[(rho*accu_width) + theta]; 
				for(int ii=-4;ii<=4;ii++)  
				{  
					for(int jj=-4;jj<=4;jj++)  
					{  
						if( (ii+rho>=0 && ii+rho<accu_height) && (jj+theta>=0 && jj+theta<accu_width) )  
						{  
							if( accumulators[((rho+ii)*accu_width) + (theta+jj)] > max )  
							{
								max = accumulators[((rho+ii)*accu_width) + (theta+jj)];
							}  
						}  
					}  
				}  

				if(max == accumulators[(rho*accu_width) + theta]) //local maxima
				{
					int x1, y1, x2, y2;  
					x1 = y1 = x2 = y2 = 0;  

					if(theta >= 45 && theta <= 135)  
					{
						if (theta>90) {
							//y = (r - x cos(t)) / sin(t)  
							x1 = width/2;  
							y1 = ((float)(rho-(accu_height/2)) - ((x1 - (width/2) ) * cos_table[theta])) / sin_table[theta] + (height / 2);
							x2 = width;  
							y2 = ((float)(rho-(accu_height/2)) - ((x2 - (width/2) ) * cos_table[theta])) / sin_table[theta] + (height / 2);  
						} else {
							//y = (r - x cos(t)) / sin(t)  
							x1 = 0;  
							y1 = ((float)(rho-(accu_height/2)) - ((x1 - (width/2) ) * cos_table[theta])) / sin_table[theta] + (height / 2);
							x2 = width*2/5;  
							y2 = ((float)(rho-(accu_height/2)) - ((x2 - (width/2) ) * cos_table[theta])) / sin_table[theta] + (height / 2); 
						}
					} else {
						//x = (r - y sin(t)) / cos(t);  
						y1 = 0;  
						x1 = ((float)(rho-(accu_height/2)) - ((y1 - (height/2) ) * sin_table[theta])) / cos_table[theta] + (width / 2);  
						y2 = height;  
						x2 = ((float)(rho-(accu_height/2)) - ((y2 - (height/2) ) * sin_table[theta])) / cos_table[theta] + (width / 2);  
					}
					x1_lines[*lines] = x1;
					y1_lines[*lines] = y1;
					x2_lines[*lines] = x2;
					y2_lines[*lines] = y2;
					(*lines)++;
				}
			}
		}
	}
}

void line_asist_GPU(uint8_t *im, int height, int width,
	float *sin_table, float *cos_table, 
	uint32_t *accum, int accu_height, int accu_width,
	int *x1, int *x2, int *y1, int *y2, int *nlines)
{
	int threshold;
	
	//GPU
	uint8_t *imageBW_GPU;
	cudaMalloc((uint8_t **)&imageBW_GPU,sizeof(uint8_t)*width*height );
	cudaMemcpy(imageBW_GPU,im,sizeof(uint8_t)*width*height,cudaMemcpyHostToDevice);
	
	uint8_t *imageOUT = (uint8_t *)malloc(sizeof(uint8_t)*width*height);
	
	float *NR_GPU;
	float *G_GPU;
	float *phi_GPU;
	float *Gx_GPU;
	float *Gy_GPU;
	uint8_t *pedge_GPU;
	uint8_t*imageOUT_GPU;

	cudaMalloc((float**)&NR_GPU,sizeof(float)*width*height);
	cudaMalloc((float**)&G_GPU,sizeof(float)*width*height);
	cudaMalloc((float**)&phi_GPU,sizeof(float)*width*height);
	cudaMalloc((float**)&Gx_GPU,sizeof(float)*width*height);
	cudaMalloc((float**)&Gy_GPU,sizeof(float)*width*height);
	cudaMalloc((uint8_t**)&pedge_GPU,sizeof(uint8_t)*width*height);
	cudaMalloc((uint8_t**)&imageOUT_GPU,sizeof(uint8_t)*width*height);

	//CANNY
	cannyGPU(imageBW_GPU, imageOUT_GPU, NR_GPU, G_GPU, phi_GPU, Gx_GPU, Gy_GPU, pedge_GPU, 1000.0, height, width);
	
	cudaMemcpy(imageOUT,imageOUT_GPU,sizeof(uint8_t)*width*height,cudaMemcpyDeviceToHost);
	//hough transform 
	houghtransform(imageOUT, width, height, accum, accu_width, accu_height, sin_table, cos_table);

	// WRITE IMAGE
	if (width>height) threshold = width/6;
	else threshold = height/6;


	getlines(threshold, accum, accu_width, accu_height, width, height, 
		sin_table, cos_table,
		x1, y1, x2, y2, nlines);
}
