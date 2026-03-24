#pragma once
#include "Macro.h"
#include "Box.h"
#include <vector>
#include <iostream>

inline int calc_blocks(int NUM) {
	int res;
	if (NUM % THREADS_PERBLOCK == 0)
		res = NUM / THREADS_PERBLOCK;
	else
		res = (NUM / THREADS_PERBLOCK) + 1;
	return res;
}

__device__ inline void check_cell_pos_overflow(int* cell_pos, int cellsPerDim)
{
	if (*cell_pos < 0)
	{
		*cell_pos += cellsPerDim;
	}
	else if (*cell_pos >= cellsPerDim)
	{
		*cell_pos -= cellsPerDim;
	}
}

__host__ __device__ inline void mymodify(float* pos, int *periodic, int boxl) {
	if (*pos < 0) {
		*pos += boxl;
		if(periodic) *periodic -= 1;
	}
	else if (*pos >= boxl) {
		*pos -= boxl;
		if(periodic) *periodic += 1;
	}
}

__host__ __device__ inline float calcDist_Squared(float* dist) {
	float ret = 0;
	for (int i = 0; i < DIMSIZE; i++) {
		ret += dist[i] * dist[i];
	}
	return ret;
}

__device__ inline void normlize(float* ori) {
	float r2 = 0;
	for (int i = 0; i < DIMSIZE; i++) {
		r2 += ori[i] * ori[i];
	}

	for (int i = 0; i < DIMSIZE; i++) {
		ori[i] /= sqrtf(r2);
	}
}

__device__ inline void modify_distance(float* dis, float boxl) {
	if (*dis >= boxl / 2.0f) *dis -= boxl;
	else if (*dis < -boxl / 2.0f) *dis += boxl;
}

__device__ inline void get_reduced_distance(float* post, float* posct, float dist[DIMSIZE], Box* box_) {
#pragma  unroll
	for (int i = 0; i < DIMSIZE; i++)
	{
		dist[i] = posct[i] - post[i];
		if (dist[i] >= box_->d[i] / 2.0f) dist[i] -= box_->d[i];
		else if (dist[i] < -box_->d[i] / 2.0f) dist[i] += box_->d[i];
	}
}

__device__ inline void getBondDistance(float* center, float* other, int* perc, int* pero, float dist[DIMSIZE], Box* box_) {
#pragma unroll
	for (int i = 0; i < DIMSIZE; i++) {
		dist[i] = (other[i] - center[i]) + (pero[i] - perc[i])*box_->d[i];
	}
}

__device__ inline void getUnreduceDistance(float* post, float* posct, float dist[DIMSIZE]) {
#pragma  unroll
	for (int i = 0; i < DIMSIZE; i++)
	{
		dist[i] = posct[i] - post[i];
		//if (dist[i] >= box.d[i] / 2.0f) dist[i] -= box.d[i];
		//else if (dist[i] < -box.d[i] / 2.0f) dist[i] += box.d[i];
	}
}

template<typename T>
__inline__ void TOGPU(T* cpu, T* gpu_) {
	cudaMemcpy(gpu_, cpu, sizeof(T), cudaMemcpyHostToDevice);
}