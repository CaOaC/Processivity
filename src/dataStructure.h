#pragma once
#include "Macro.h"
#include "var.h"

struct V3 {
	float e[DIMSIZE];
};

struct Prop
{
	float pos[DIMSIZE];
	float dposkmc[DIMSIZE];
	float force[DIMSIZE];
	int periodic[DIMSIZE];
	float pe[2];
	bool IsActive;
	float kappa;
	int kickNumber;
};

struct Adj {
	int Cij;
	int io;
};

struct KMC {

	V3* kickpos_;
	V3* kickdpos_;
	float* rate_;
	float* rate;
	float* dU_;
	int* flag_;
	int* flag;

	float l = kick::l;
	float s = kick::s0;
	float kappa = kick::kappa_short1;
	float kappa_long = kick::kappa_long;
	float ks = kick::ks;
	float ksb = kick::ksb;
	float klb = kick::klb;
	double totalKmcTime = 0;
	void memory();
};

struct Rg {
	float centerPos[DIMSIZE];
	float* dis_sq_;
	float Rg;
	float* Rg_;

	void memory();
};

struct ContactMap {
	int* CM_ = nullptr;
	int* CM = nullptr;

	float* dis_ = nullptr;
	float* dis = nullptr;

	void memory();
	void deviceToHost();
};

struct Para {

	float beta = media::beta;

	float mu = media::mu;

	float dt = sim::dt;
};