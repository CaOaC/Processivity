#pragma once
#include <random>

namespace var
{
	//Particle
	extern unsigned int seed; // random number

	//Box
	extern float boxlength;

	//host rand engine
	extern std::default_random_engine e;
	extern std::uniform_real_distribution<float> u;
};

namespace media {
	extern float beta;
	extern float mu;
}

namespace rouse {
	extern float b;
	extern float k;
}

namespace lj {
	extern float sigma;
	extern float epi;
	extern float cutoff;

	extern float Ecut;
	extern float ro;
	//another part
	extern float rc;
	extern float rb;
	extern float barrier;
}

namespace bend {
	extern float ka;
}

namespace fene {
	extern float kb;
	extern float R0;
}

namespace kick {
	extern float kappa_short1;
	extern float kappa_short2;
	//extern float kappa;
	extern float kappa_long;
	extern float ksb;
	extern float klb;
	extern float ks;
	extern float l;
	extern float s0;
	extern float p10;
	extern float p_slide;
}

namespace sim {
	extern int kickTime;
	extern int warm_cycles;
	extern float dt;
	extern int totalnumber;
	extern int activenumber;
	extern int ensembleID;
	extern int stepsPersecond;
	extern bool isRandConfig;
	extern bool isCofRij;
	extern bool isInputFile;
	__device__ __constant__ extern int totalnumber_;
	__device__ __constant__ extern int activenumber_;
}