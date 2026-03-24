#include "var.h"

namespace media {
	float beta = 1.0;
	float mu = 0.1;
}

namespace rouse {
	float b = 1.0;
	float k = 3 / (media::beta * b * b);
}

namespace lj {
	float sigma = 1.0;
	float epi = 1.0/(media::beta);
	float cutoff = 1.12 * sigma;

	float Ecut = 10.0 * epi;
	float ro = 0.885 * sigma;

	//another part
	float rc = 0.8 * sigma;
	float rb = 0.4 * sigma;
	float barrier = 10.0/(media::beta);
}

namespace bend {
	float ka = 10 * lj::epi;
}

namespace fene {
	float kb = 30 / (media::beta * lj::sigma * lj::sigma);
	float R0 =  2.0 * lj::sigma;
}

namespace kick {
	float kappa_short1 = 0.4;
	float kappa_short2 = 0.4;
	float kappa_long = 0.4;
	float ksb = 1.0;
	float klb = 1.0;
	float ks = 0;
	float l = 0.005*lj::sigma;
	float s0 = 1.0;
	float p10 = 0.0;
	float p_slide = 0.9;
}

namespace var {
	//Particle
	unsigned int seed = 114514 + sim::ensembleID; // random number
	//Box
	float boxlength = 150 * lj::sigma;
	//host rand engine
	std::default_random_engine e;
	std::uniform_real_distribution<float> u(0, 1.0);
}

namespace sim {
	int kickTime = 2e5;
	int warm_cycles = 100;
	float dt = 1.0e-4 / (media::mu);
	int totalnumber = 1500;
	int activenumber = 1500;
	int ensembleID = 0;
	int stepsPersecond = int(1.0/dt);
	bool isRandConfig = true;
	bool isCofRij = true;
	bool isInputFile = false;
	__device__ __constant__ int totalnumber_;
	__device__ __constant__ int activenumber_; 
}