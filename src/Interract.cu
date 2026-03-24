#include "Interract.h"
#include <math.h>

Interract::Interract()
{
}

LJ::LJ():Interract()
{
}

LJ::LJ(float sigma, float epsion, float cutoff)
{
	this->sigma = sigma;
	this->epsion = epsion;
	this->cutoff = cutoff;
}

__host__ __device__ float LJ::calcForce(float r)
{
	float ret = 0;
	if ( r < cutoff ) {
		float r6 = powf(sigma / r, 6);
		float r12 = r6 * r6;
		ret = 48 * epsion / r * (r12 - 0.5*r6);
	}
	else {
		ret = 0;
	}
	return ret;
}

__device__ float LJ::calcPE(float r)
{
	float ret = 0;
	if (r < cutoff) {
		float r6 = powf(sigma / r, 6);
		float r12 = r6 * r6;
		ret = 4 * epsion * (r12 - r6 + 0.25);
	}
	return ret;
}

HARMONIC::HARMONIC()
{
}

HARMONIC::HARMONIC(float k, float re)
{
	this->k = k;
	this->re = re;
}

__host__ __device__ float HARMONIC::calcForce(float r)
{
	return -k*(r-re);
}

__host__ __device__ float HARMONIC::calcPE(float r)
{
	return 0.5*k*(r-re)*(r-re);
}

SPHRERCONFIE::SPHRERCONFIE()
{
}

SPHRERCONFIE::SPHRERCONFIE(float k, float rc)
{
	this->k = k;
	this->rc = rc;
}

SPHRERCONFIE::~SPHRERCONFIE()
{
}

__host__ __device__ float SPHRERCONFIE::calcForce(float r)
{
	float ff = 0;

	if (r > rc) {
		ff = -k * r;
	}

	return ff;
}

FENE::FENE()
{
}

FENE::FENE(float kb, float R0)
{
	this->R0 = R0;
	this->kb = kb;
}

__host__ __device__ float FENE::calcForce(float r)
{
	float ff = 0;

	if (r <= R0) {
		ff = - kb * r / (1 - (r / R0) * (r / R0));
	}

	return ff;
}

__device__ float FENE::calcPE(float r)
{
	float u = 0;

	if (r <= R0) {
		u = -0.5 * kb * R0 * R0 * log(1 - (r / R0) * (r / R0));
		//printf("u %f\n", u);
	}
	return u;
}

Kratky_Porod::Kratky_Porod()
{
}

Kratky_Porod::Kratky_Porod(float ka)
{
	this->ka = ka;
}

__device__ float Kratky_Porod::calcPE(float costheta)
{
	return ka * (1 - costheta);
}

SC::SC()
{
}

SC::SC(float sigma, float epsion, float Ecut, float ro, float cutoff)
{
	this->sigma = sigma;
	this->epsion = epsion;
	this->Ecut = Ecut;
	this->ro = ro;
	this->cutoff = cutoff;

	//second part for peter
	
}

SC::SC(float sigma, float epsion, float Ecut, float ro, float cutoff, float rc, float rb, float barrier)
{
	this->sigma = sigma;
	this->epsion = epsion;
	this->Ecut = Ecut;
	this->ro = ro;
	this->cutoff = cutoff;

	this->barrier = barrier;
	this->rb = rb;
	this->rc = rc;
	this->k = (Ecut - barrier) / (rc - rb);
}


__device__ float SC::calcForce(float r)
{
	float ret = 0;
	if (r < cutoff) {
		float r6 = powf(sigma / r, 6);
		float r12 = r6 * r6;

		if (r < rb) {
			ret = 0;
		}else if(r < rc) {
			ret = -k;
		}
		else if (r < ro) {
			ret = 48 * epsion / r * (r12 - 0.5 * r6) / powf(coshf(2 * 4 / Ecut * epsion * (r12 - r6 + 0.25) - 1), 2);
			//ret = 0;
		}
		else {
			//ret = 0;
			ret = 48 * epsion / r * (r12 - 0.5 * r6);
		}
		/*
		if (r < ro) {
			ret = 48 * epsion / r * (r12 - 0.5 * r6) /powf(coshf(2*4/Ecut*epsion*(r12 - r6 + 0.25) - 1), 2);
			//ret = 0;
		}
		else {
			ret = 48 * epsion / r * (r12 - 0.5 * r6);
			//ret = 0;
		}
		*/
		
	}
	return ret;
}

__device__ float SC::calcPE(float r)
{
	float ret = 0;
	if (r < cutoff) {
		float r6 = powf(sigma / r, 6);
		float r12 = r6 * r6;
		float Ulj = 4 * epsion * (r12 - r6 + 0.25);
		
		if (r < rb) {
			ret = barrier;
		}
		else if (r < rc) {
			ret = k*r + (Ecut*rb - barrier*rc)/(rb-rc);
		}
		else if (r < ro) {
			ret = 0.5 * Ecut * (1 + tanhf(2 * Ulj / Ecut - 1));
		}
		else {
			ret = Ulj;
		}
		/*
		if (r < ro) {
			ret = 0.5 * Ecut * (1 + tanhf(2 * Ulj / Ecut - 1));
			//printf("pe %f\n", ret);
		}
		else {
			ret = Ulj;
		}
		*/
		
	}
	return ret;
}
