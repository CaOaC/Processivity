#include "System.h"

System::System()
{
}

System::~System()
{
}

void System::memory()
{
	cudaError_t err = cudaMallocManaged((void**)&isDivergencyed, sizeof(int));
}

__global__ void CalcDistance(Prop* prop_, float* dis_) {
	int tid = threadIdx.x + blockDim.x * blockIdx.x;

	if (tid < sim::activenumber_ * sim::activenumber_) {

		int i = tid / sim::activenumber_;
		int j = tid % sim::activenumber_;

		float dist[3] = { 0,0,0 };
		getUnreduceDistance(prop_[i].pos, prop_[j].pos, dist);
		float r2 = 0;

		for (int k = 0; k < DIMSIZE; k++) {
			r2 += dist[k] * dist[k];
		}
		float r = sqrtf(r2);

		dis_[tid] = r;

		//printf("r: %f %d %d %f %f %f \n", r, i, j, dist[0], dist[1], dist[3]);
	}
}