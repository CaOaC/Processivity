#pragma once
#include "Macro.h"
#include "dataStructure.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "var.h"

class PullMethod
{
public:
	float pf_ = 0.0;
	V3* dir_;

	void memory();
	void calcDirection(Prop* prop_);
	void addPullForce(Prop* prop_);
};

__global__ void CalcDirection(Prop* prop_, PullMethod pobject_);
__global__ void AddPullForce(Prop* prop_, PullMethod pobject_);

