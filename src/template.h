#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "dataStructure.h"
#include "Cell.h"

template<typename BOND, typename BOND1>
__global__ void CalcBonddU(Prop* prop_, KMC* kmc_, BOND bond, BOND1 bond1, Box* box, curandState* states_) {
	int tid = threadIdx.x + blockDim.x * blockIdx.x;

	if (tid < sim::activenumber_ * sim::activenumber_) {
		int x = tid / sim::activenumber_;
		int y = tid % sim::activenumber_;

		if (x != y) {

			float uxy = 0; float uxy_ = 0;
			float ux = 0; float ux_ = 0;
			float uy = 0; float uy_ = 0;

			if (abs(x - y) == 1) {
				uxy = getBondPe(prop_[x].pos, prop_[y].pos, box, bond, bond1);
				uxy_ = getBondPe(kmc_->kickpos_[tid].e, kmc_->kickpos_[y*sim::activenumber_ + x].e, box, bond, bond1);
			}

			ux = getUnkcikBondPE(prop_, x, prop_[x].pos, box, bond, bond1);
			ux_ = getKcikBondPE(prop_, x, y, kmc_->kickpos_[tid].e, kmc_->kickpos_[y * sim::activenumber_ + x].e, box, bond, bond1);

			uy = getUnkcikBondPE(prop_, y, prop_[y].pos, box, bond, bond1);
			uy_ = getKcikBondPE(prop_, y, x, kmc_->kickpos_[y*sim::activenumber_+x].e, kmc_->kickpos_[tid].e, box, bond, bond1);

			float du = (ux_ - ux) + (uy_ - uy) - (uxy_ - uxy);

			kmc_->dU_[tid] = du;

			//printf("du %f\n", du);
			//printf("u:%f %f %f %f %d %d\n", ux, ux_, uy, uy_, x, y);
		}
	}
}

template<typename NOBOND>
__global__ void CalcNoBondU1(Prop* prop_, KMC* kmc_, NOBOND nobond, Box* box, curandState* states_) {
	int tid = threadIdx.x + blockDim.x * blockIdx.x;

	if (tid < sim::activenumber_ * sim::activenumber_) {
		int i = tid / sim::activenumber_;
		int j = tid % sim::activenumber_;

		if (abs(i - j) >= 2) {
			float dist[DIMSIZE];

			get_reduced_distance(prop_[j].pos, prop_[i].pos, dist, box);

			float r2 = 0;
			for (int k = 0; k < DIMSIZE; k++) {
				r2 += dist[k] * dist[k];
			}
			float r = sqrtf(r2);

			//printf("r %f\n", r);

			float u = nobond.calcPE(r);

			//if (u > 0) {
			//	printf("tttt:%f\n", u);
			//}
			atomicAdd(&prop_[i].pe[0], u);
		}
	}
}

template <typename NOBOND>
__global__ void CalcNoBondU2(Cell c, Prop* prop_, KMC* kmc_, NOBOND nobond, Box* box, curandState* states_) {
	int tid = threadIdx.x + blockDim.x * blockIdx.x;

	if (tid < sim::activenumber_ * sim::activenumber_) {
		//i,j represet atom is pulled.
		int x = tid / sim::activenumber_;
		int y = tid % sim::activenumber_;
	
		if (abs(x - y) >= 2) {
			
			float uxy = 0; float uxy_ = 0;
			float ux = 0; float ux_ = 0;
			float uy = 0; float uy_ = 0;

			uxy = getPairPe(prop_[x].pos, prop_[y].pos, box, nobond);
			uxy_ = getPairPe(kmc_->kickpos_[tid].e, kmc_->kickpos_[y * sim::activenumber_ + x].e, box, nobond);
			ux = prop_[x].pe[0]; uy = prop_[y].pe[0];

			ux_ = getKickNobondPE(c, prop_, x, y, kmc_->kickpos_[tid].e, kmc_->kickpos_[y * sim::activenumber_ + x].e, box, nobond);

			uy_ = getKickNobondPE(c, prop_, y, x, kmc_->kickpos_[y * sim::activenumber_ + x].e, kmc_->kickpos_[tid].e, box, nobond);

			float du = (ux_ - ux) + (uy_ - uy) - (uxy_ - uxy);

			//if (du < 10) {
			//	printf("xy %d %d\n", x, y);
			//}

			//printf("u:%f %f %f %f %f %f\n", ux, ux_, uy, uy_, uxy, uxy_);
			kmc_->dU_[tid] += du;
			//printf("%f\n", kmc_->dU_[tid]);
		}
	}
}

template <typename BOND, typename BOND1>
__global__ void Bond_force_left(Prop* prop_, Box* box_, BOND fene, BOND1 bond1) {
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid < sim::totalnumber_ && tid > 0) {
		float r2 = 0;
		float dist[DIMSIZE];
		float r;
		float ff;

		//get_reduced_distance(prop_[tid - 1].pos, prop_[tid].pos, dist, box_);
		getBondDistance(prop_[tid - 1].pos, prop_[tid].pos, prop_[tid - 1].periodic, prop_[tid].periodic, dist, box_);

		for (int j = 0; j < DIMSIZE; j++) {
			r2 += dist[j] * dist[j];
		}
		r = sqrtf(r2);

		ff = fene.calcForce(r);

		float ff1 = bond1.calcForce(r);

		ff = ff + ff1;
		//printf("%f %f\n", r, ff);

		for (int j = 0; j < DIMSIZE; j++) {
			prop_[tid].force[j] += ff * dist[j] / r;
		}
	}
}

template <typename BOND, typename BOND1>
__global__ void Bond_force_right(Prop* prop_, Box* box_, BOND fene, BOND1 bond1) {
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid < sim::totalnumber_ - 1) {
		float r2 = 0;
		float dist[DIMSIZE];
		float r;
		float ff;

		//get_reduced_distance(prop_[tid + 1].pos, prop_[tid].pos, dist, box_);
		getBondDistance(prop_[tid + 1].pos, prop_[tid].pos, prop_[tid + 1].periodic, prop_[tid].periodic, dist, box_);
		for (int j = 0; j < DIMSIZE; j++) {
			r2 += dist[j] * dist[j];
		}

		r = sqrtf(r2);
		ff = fene.calcForce(r);
		float ff1 = bond1.calcForce(r);

		ff = ff + ff1;
		for (int j = 0; j < DIMSIZE; j++) {
			prop_[tid].force[j] += ff * dist[j] / r;
		}
	}
}


template <typename NOBOND>
__device__ float getPairPe(float* pos1, float* pos2, Box* box, NOBOND& nobond) {
	float dist[DIMSIZE];
	float r, r2;
	float u = 0;
	if (pos1 != pos2) {
		r2 = 0;
		get_reduced_distance(pos1, pos2, dist, box);
		for (int i = 0; i < DIMSIZE; i++) {
			r2 += dist[i] * dist[i];
		}
		r = sqrtf(r2);
		u = nobond.calcPE(r);
	}
	return u;
}

template <typename BOND, typename BOND1>
__device__ float getBondPe(float* pos1, float* pos2, Box* box, BOND& bond, BOND1& bond1) {
	float dist[DIMSIZE];
	float r, r2;
	float u = 0;
	float u1 = 0; float u2 = 0;
	if (pos1 != pos2) {
		r2 = 0;
		get_reduced_distance(pos1, pos2, dist, box);
		for (int i = 0; i < DIMSIZE; i++) {
			r2 += dist[i] * dist[i];
		}
		r = sqrtf(r2);
		//printf("r %f\n", r);
		u1 = bond.calcPE(r);
		u2 = bond1.calcPE(r);
		u = u1 + u2;
		//printf("%f %f\n", u1, u2);
	}
	return u;
}

template <typename BOND, typename BOND1>
__device__ float getUnkcikBondPE(Prop* prop_, int tid, float* pos, Box* box, BOND& harmonic, BOND1& bond1)
{
	float dist[DIMSIZE];
	float r, r2;
	float u1 = 0;
	float u2 = 0;
	//ul
	if (tid - 1 >= 0) {
		r = 0; r2 = 0;
		get_reduced_distance(prop_[tid - 1].pos, pos, dist, box);

		for (int i = 0; i < DIMSIZE; i++) {
			r2 += dist[i] * dist[i];
		}
		r = sqrtf(r2);
		u1 = harmonic.calcPE(r) + bond1.calcPE(r);
	}

	//ur
	if (tid + 1 < sim::totalnumber_) {
		r = 0; r2 = 0;
		get_reduced_distance(prop_[tid + 1].pos, pos, dist, box);
		for (int i = 0; i < DIMSIZE; i++) {
			r2 += dist[i] * dist[i];
		}
		r = sqrtf(r2);
		//u2 = harmonic.calcPE(r);
		u2 = harmonic.calcPE(r) + bond1.calcPE(r);
		//printf("%f %f\n", harmonic.calcPE(r), bond1.calcPE(r));
	}

	return u1 + u2;
}

template <typename BOND, typename BOND1>
__device__ float getKcikBondPE(Prop* prop_, int tid, int y, float* pos, float* posy, Box* box, BOND& harmonic, BOND1& bond1)
{
	//ul
	float u1 = 0; float u2 = 0;
	if (tid - 1 >= 0) {
		if (tid - 1 == y) {
			u1 = getBondPe(pos, posy, box, harmonic, bond1);
		}
		else {
			u1 = getBondPe(prop_[tid - 1].pos, pos, box, harmonic, bond1);
		}

	}

	//ur
	if (tid + 1 < sim::totalnumber_) {
		if (tid + 1 == y) {
			u2 = getBondPe(pos, posy, box, harmonic, bond1);
		}
		else {
			u2 = getBondPe(prop_[tid + 1].pos, pos, box, harmonic, bond1);
		}

	}

	//printf("u1+u2 %f\n", u1 + u2);
	return (u1+u2);
}

template <typename NOBOND>
__device__ float getKickNobondPE(Cell c, Prop* prop_, int x, int y, float* posx, float* posy, Box* box, NOBOND nobond) {
	float u = 0; float temp = 0;
	int cellid[DIMSIZE];

	for (int i = 0; i < 27; i++) {
		cellid[2] = i / (DIMSIZE * DIMSIZE);
		cellid[0] = i % (DIMSIZE * DIMSIZE) / DIMSIZE;
		cellid[1] = i % (DIMSIZE * DIMSIZE) % DIMSIZE;

		for (int j = 0; j < DIMSIZE; j++) {
			cellid[j] += (c.cell_list[x].cellid[j] - 1);
			//if (cellid[i] < 0 || cellid[i] > c.dim[i]) return;
			check_cell_pos_overflow(cellid + j, c.dim[j]);
			//printf("%d %d %d\n", cellid[0], cellid[1], cellid[2]);
		}

		int cell_id = cellid[0] * c.dim[1] + cellid[1] + cellid[2] * c.dim[0] * c.dim[1];

		int tid_compare = c.cell_head[cell_id];

		while (tid_compare != -1)
		{
			temp = 0;
			if (tid_compare > sim::totalnumber_) printf("wrong %d\n", tid_compare);
			if (abs(tid_compare - x) < 2) {
				//printf("dadadad\n");
				tid_compare = c.cell_list[tid_compare].next;
				continue;
			}
			else if(tid_compare == y){
				temp = getPairPe(posx, posy, box, nobond);
			}
			else {
				temp = getPairPe(posx, prop_[tid_compare].pos, box, nobond);
			}
			tid_compare = c.cell_list[tid_compare].next;
			u += temp;
		}
	}
	return u;
}

template <typename BOND>
__global__ void CalcPE(Prop* prop_, Box* box, BOND bond) {
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid >= sim::totalnumber_) return;

	prop_[tid].pe[0] = getUnkcikBondPE(prop_, tid, prop_[tid].pos, box, bond);
}