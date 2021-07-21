#ifndef ROUTINESGPU_H
#define ROUTINESGPU_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" 
#endif
void line_asist_GPU(uint8_t *im, int height, int width,
	float *sin_table, float *cos_table, 
	uint32_t *accum, int accu_height, int accu_width,
	int *x1, int *y1, int *x2, int *y2, int *nlines);

#endif

