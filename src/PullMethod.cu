#include "PullMethod.h"

void PullMethod::memory()
{
	cudaMalloc((void**)&dir_, sizeof(V3));
}

void PullMethod::calcDirection(Prop* prop_)
{
    CalcDirection << <1, 100 >> > (prop_, *this);
}

void PullMethod::addPullForce(Prop* prop_)
{
    AddPullForce << <1, 100 >> > (prop_, *this);
}

__global__ void AddPullForce(Prop* prop_, PullMethod pobject_) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < 100) {
        for (int i = 0; i < DIMSIZE; i++) {
            prop_[tid].force[i] += (-pobject_.pf_ * pobject_.dir_[0].e[i]);
        }
    }
}

__global__ void CalcDirection(Prop* prop_, PullMethod pobject_)
{
	int tid = threadIdx.x;
	__shared__ float sharedData[3*100];

	if (tid < 100) {
		for (int i = 0; i < DIMSIZE; i++) {
			sharedData[tid*3+i] = prop_[sim::totalnumber_ - tid - 1].pos[i] - prop_[tid].pos[i];
		}

		__syncthreads();

        // 计算差值的平均
        if (tid == 0) {
            float avgDifference[3] = { 0.0f, 0.0f, 0.0f }; // 存储平均差值

            // 累加所有的差值
            for (int i = 0; i < 100; i++) {
                for (int j = 0; j < DIMSIZE; j++) {
                    avgDifference[j] += sharedData[i * 3 + j]; // 累加差值
                }
            }

            // 计算平均差值
            for (int j = 0; j < DIMSIZE; j++) {
                avgDifference[j] /= 100.0f; // 计算平均
            }

            // 计算单位向量
            float magnitude = 0.0f;
            for (int j = 0; j < DIMSIZE; j++) {
                magnitude += avgDifference[j] * avgDifference[j]; // 计算模的平方
            }
            magnitude = sqrtf(magnitude); // 计算模

            // 归一化为单位向量
            float unitVector[3];
            if (magnitude > 0) { // 防止除以零
                for (int j = 0; j < DIMSIZE; j++) {
                    pobject_.dir_[0].e[j] = avgDifference[j] / magnitude; // 归一化
                }
                //printf("%f %f %f\n", pobject_.dir_[0].e[0], pobject_.dir_[0].e[1], pobject_.dir_[0].e[2]);
            }
            else {
                // 如果模为零，可以设置 unitVector 为零向量
                for (int j = 0; j < DIMSIZE; j++) {
                    pobject_.dir_[0].e[j] = 0.0f;
                }
            }
            // 在这里可以将 avgDifference 保存或用于后续计算
            // 例如，可以将其写回全局内存
        }
	}
}

