#pragma once
#include "var.h"
#include "device_launch_parameters.h"

class Interract
{
public:
	Interract();
};

class LJ :public Interract{
public:
	float sigma;
	float cutoff;
	float epsion;
public:
	LJ();
	LJ(float sigma, float epsion, float cutoff);
	__host__ __device__ float calcForce(float r);
	__host__ __device__ float calcPE(float r);
};

class SC : public Interract {
public:
	float sigma;
	float epsion;
	float Ecut;
	float ro;
	float cutoff;
	// another part
	float rc;
	float rb;
	float barrier;
	float k;
public:
	SC();
	SC(float sigma, float epsion, float Ecut, float ro, float cutoff);
	SC(float sigma, float epsion, float Ecut, float ro, float cutoff,
		float rc, float rb, float barrier);
	__host__ __device__ float calcForce(float r);
	__host__ __device__ float calcPE(float r);
};

class FENE : public Interract {
public:
	float kb;
	float R0;
public:
	FENE();
	FENE(float kb, float R0);
	__host__ __device__ float calcForce(float r);
	__host__ __device__ float calcPE(float r);
};

class Kratky_Porod : public Interract {
public:
	float ka;
public:
	Kratky_Porod();
	Kratky_Porod(float ka);
	__host__ __device__ float calcPE(float theta);
};

class HARMONIC :public Interract {
public:
	float k;
	float re;

public:
	HARMONIC();
	HARMONIC(float k, float re);
	__host__ __device__ float calcForce(float r);
	__host__ __device__ float calcPE(float r);
};

class SPHRERCONFIE :public Interract {
public:
	float rc;
	float k;

public:
	SPHRERCONFIE();
	SPHRERCONFIE(float k, float rc);
	~SPHRERCONFIE();
	__host__ __device__ float calcForce(float r);
};