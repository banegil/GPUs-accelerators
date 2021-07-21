// pimienta.cl
// Kernel source file for remove_noise

#define MAX_WINDOW_SIZE 5*5


float fabsf(float value){
	return (value>0)? value : -value;
}

void swap ( float* a, float* b )
{
    float tmp = *a;
    *a = *b;
    *b = tmp;
}

void buble_sort(float array[], int size)
{
	int i, j;
	float tmp;

	for (i=1; i<size; i++)
		for (j=0 ; j<size - i; j++)
			if (array[j] > array[j+1]){
				tmp = array[j];
				array[j] = array[j+1];
				array[j+1] = tmp;
			}
}


__kernel
void remove_noise(__global float *im, __global float *image_out, 
	const float thredshold, const int window_size,
	const int height, const int width)
{
	int ii, jj;

	float window[MAX_WINDOW_SIZE];
	float median;
	int ws2 = (window_size-1)>>1; 
	
	int idx = get_global_id(0);
	int idy = get_global_id(1);
	
	if((idx >= ws2) && (idx < width-ws2))
		if((idy >= ws2) && (idy < height-ws2))
		{
			for (ii =-ws2; ii<=ws2; ii++)
				for (jj =-ws2; jj<=ws2; jj++)
					window[(ii+ws2)*window_size + jj+ws2] = im[(idx+ii)*width + idy+jj];

			// SORT

			buble_sort (window, window_size*window_size);

			median = window[(window_size*window_size-1)>>1+1];

			if (fabs((median-im[idx*width+idy])/median) <=thredshold)
				image_out[idx*width + idy] = im[idx*width+idy];
			else
				image_out[idx*width + idy] = median;

				
		}
}

