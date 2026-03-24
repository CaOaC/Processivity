#pragma once
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include "curand_kernel.h"

#include <stdio.h>
#include "Macro.h"
#include "var.h"
#include "toolsfunc.h"
#include "Box.h"
#include "Interract.h"

#include <vector>
#include <iostream>

#include "dataStructure.h"

class Particle
{
public:

	Prop* prop_ = nullptr;
	Prop* prop = nullptr;

	float* dis = nullptr;
	float* dis_ = nullptr;

	ContactMap* cm;
	ContactMap* cm_;

	Para para;

	//Rg rg;
	KMC* kmc;
	KMC* kmc_;

	unsigned int seed = var::e();
	curandState* states;

public:
	Particle();
	~Particle();
	void init(Box& box);
	void initProp(Box& box, bool isRandConfig, bool isInputFile);
	void memory();
	void curandinit();
	void contactMap(Box* box_);
	void no_bond_force(Box* box_, LJ& inter);
	void clear_force();
	void updatePosition();
	void distanceMap(Box* box_);
	int get_bondnumber();
	void calcSphereConfineForce(Box* box_, SPHRERCONFIE&);
	void calcBendingForce(float ka);
	void updateKickPosition();
	void perodicBoundary(Box* box_);

	//template<typename BOND, typename NOBOND>
	//void runKMC(BOND& bond, NOBOND& nobond, Kratky_Porod angle, Box* box_);
	//template <typename BOND, typename NOBOND>
	//void runMD(Box* box_, BOND& bond, NOBOND& lj, Cell& c);
	template <typename BOND, typename BOND1>
	void bond_force(Box* box_, BOND&, BOND1&);
	template <typename BOND>
	void calcPE(Box* box_, BOND& bond);
};

__global__ void CalcAfterKickPosition(Prop* prop_, KMC* kmc_, Box* box);
__global__ void CalcBendingdU(Prop* prop_, KMC* kmc_, Kratky_Porod angle);
__global__ void CalcRate(KMC* kmc_, Prop* prop_, curandState* state, float beta, float tau, bool isCofRij);
__global__ void CalcGrapplingIndicator(KMC* kmc_, curandState* state, float tau, float* dis_, bool isCofRij);
__global__ void IsPosDivergency(Prop* prop_, int* isDivergencyed);
__global__ void UpdateKickPosition(Prop* prop_);
//__global__ void Update_flag(KMC* kmc_, curandState* state, float p10, float p_slide);
__global__ void Update_flag(KMC* kmc_, Prop* prop_, float beta, curandState* state, float p10, float p_slide);


template<typename BOND>
inline void Particle::calcPE(Box* box_, BOND& bond)
{
	CalcPE << <calc_blocks(sim::totalnumber), THREADS_PERBLOCK >> > (prop_, box_, bond);
	cudaMemcpy(prop, prop_, sizeof(Prop) * sim::totalnumber, cudaMemcpyDeviceToHost);
	float totalPE = 0;
	for (int i = 0; i < sim::totalnumber; i++) {
		totalPE += prop[i].pe[0];
	}
	printf("totalPE is %f\n", totalPE);
}

template <typename BOND, typename BOND1>
inline void Particle::bond_force(Box* box_, BOND& bond, BOND1& bond1)
{
	Bond_force_left << <calc_blocks(sim::totalnumber), THREADS_PERBLOCK >> > (prop_, box_, bond, bond1);
	Bond_force_right << <calc_blocks(sim::totalnumber), THREADS_PERBLOCK >> > (prop_, box_, bond, bond1);
}

/*
template<typename BOND, typename NOBOND>
void Particle::runKMC(BOND& bond, NOBOND& nobond, Kratky_Porod angle, Box* box)
{
	CalcNoBondU1 << <calc_blocks(sim::activenumber * sim::activenumber), THREADS_PERBLOCK >> > (prop_, kmc_, nobond, box, states);

	do {

		CalcAfterKickPosition << <calc_blocks(sim::activenumber * sim::activenumber), THREADS_PERBLOCK >> > (prop_, kmc_, box);

		CalcBonddU << <calc_blocks(sim::activenumber * sim::activenumber), THREADS_PERBLOCK >> > (prop_, kmc_, bond, box, states);

		CalcNoBondU2 << < calc_blocks(sim::activenumber * sim::activenumber), THREADS_PERBLOCK >> > (prop_, kmc_, nobond, box, states);

		CalcBendingdU << < calc_blocks(sim::activenumber * sim::activenumber), THREADS_PERBLOCK >> > (prop_, kmc_, angle);

		cudaMemset(kmc.sumRate_, 0, sizeof(float));

		CalcRate << <calc_blocks(sim::activenumber * sim::activenumber), THREADS_PERBLOCK >> > (kmc_, media::beta);

		updateKickPosition();

		perodicBoundary(box);

		//printf("sumRate: %f\n", kmc.sumRate);

	} while (std::isinf(kmc.sumRate) || kmc.sumRate < 1e-2);

	float rd;
	float secondTerm;

	do {
		rd = var::u(var::e);
		secondTerm = -logf(rd);
	} while (std::isinf(secondTerm));

	//printf("%f %f\n", kmc.sumRate, para.mcdt);
	para.mcdt = (1.0 / kmc.sumRate) * secondTerm;
	//printf("%f\n", para.mcdt);
	kmc.totalKmcTime += para.mcdt;
}
*/

/*
template <typename BOND, typename NOBOND>
void Particle::runMD(Box* box, BOND& bond, NOBOND& nobond, Cell& c)
{
	clear_force();

	//no_bond_force(box, nobond);

	bond_force(box, bond);

	calcBendingForce(bend::ka);

	updatePosition();

	perodicBoundary(box);
}
*/