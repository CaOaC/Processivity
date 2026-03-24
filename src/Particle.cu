#include "Particle.h"

Particle::Particle()
{
}

Particle::~Particle()
{
}

void Particle::init(Box& box)
{
	memory();
	curandinit();
	initProp(box, sim::isRandConfig, sim::isInputFile);
	(*cm).memory();
	(*kmc).memory();
	TOGPU(cm, cm_);
	TOGPU(kmc, kmc_);
}

void Particle::memory()
{   
	cudaMalloc((void**)&prop_, sizeof(Prop) * sim::totalnumber);
	prop = new Prop[sim::totalnumber];

	cudaMalloc((void**)&states, sizeof(curandState) * sim::totalnumber * sim::totalnumber);
	cudaMalloc((void**)&cm_, sizeof(ContactMap));
	cudaMalloc((void**)&kmc_, sizeof(KMC));
	cm = new ContactMap();
	kmc = new KMC();

	cudaMalloc((void**)&dis_, sizeof(float) * sim::totalnumber * sim::totalnumber);
	dis = new float[sim::totalnumber * sim::totalnumber];

	//cudaMalloc((void**)&adj_, sizeof(Adj) * sim::totalnumber * sim::totalnumber);
	//adj = new Adj[N * sim::totalnumber];

	//rg.memory(sim::totalnumber);
	//cudaMalloc((void**)&rgPara_, sizeof(RgPara));
}

__global__ void CalcContactMap(Prop* prop_, ContactMap* cm_, Box* box);
void Particle::contactMap(Box* box_)
{
	CalcContactMap << <calc_blocks(sim::totalnumber * sim::totalnumber), THREADS_PERBLOCK >> > (prop_, cm_, box_);
}

__global__ void CalcContactMap(Prop* prop_, ContactMap* cm_, Box* box) {
	int offset = threadIdx.x + blockDim.x * blockIdx.x;

	int tid = offset / sim::totalnumber_;
	int tid_compare = offset % sim::totalnumber_;

	if (tid >= sim::totalnumber_) return;

	float dist[DIMSIZE];
	get_reduced_distance(prop_[tid_compare].pos, prop_[tid].pos, dist, box);

	float r2 = 0;
	for (int i = 0; i < DIMSIZE; i++) {
		r2 += dist[i] * dist[i];
	}
	//printf("%f\n", r2);
	float r = sqrtf(r2);
	if (r < 5) {
		atomicAdd(&cm_->CM_[offset], 1);
	}
}
__global__ void No_bond_force(Particle p, Box* box, LJ inter);

void Particle::no_bond_force(Box* box, LJ& inter)
{
	No_bond_force << <calc_blocks(sim::totalnumber * sim::totalnumber), THREADS_PERBLOCK >> > (*this, box, inter);
}

__global__ void No_bond_force(Particle p, Box* box, LJ inter) {
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	int i = tid / sim::totalnumber_;
	int j = tid % sim::totalnumber_;

	if (i >= sim::totalnumber_ || i==j) return;

	float dist[DIMSIZE];

	get_reduced_distance(p.prop_[j].pos, p.prop_[i].pos, dist, box);

	float r2 = 0;
	for (int k = 0; k < DIMSIZE; k++) {
		r2 += dist[k] * dist[k];
	}
	float r = sqrtf(r2);

	//printf("r %f\n", r);

	float ff = inter.calcForce(r);
	for (int k = 0; k < DIMSIZE; k++) {
		atomicAdd(&p.prop_[i].force[k], ff * dist[k] / r);
	}
}

__global__ void DistanceMap(Prop* prop_, ContactMap* cm, Box* box);

void Particle::distanceMap(Box* box)
{
	DistanceMap << <calc_blocks(sim::totalnumber * sim::totalnumber), THREADS_PERBLOCK >> > (prop_, cm_, box);
}

__global__ void DistanceMap(Prop* prop_, ContactMap* cm, Box* box) {
	int tid = threadIdx.x + blockDim.x * blockIdx.x;

	int i = tid / sim::totalnumber_;
	int j = tid % sim::totalnumber_;

	if (i >= sim::totalnumber_ || j >= sim::totalnumber_) return;

	float* posi = prop_[i].pos;
	float* posj = prop_[j].pos;

	float r2 = 0;
	for (int k = 0; k < DIMSIZE; k++) {

		r2 += powf((posi[k]+ prop_[i].periodic[k]*box->d[k] - (posj[k] + prop_[j].periodic[k] * box->d[k])), 2);
	}

	//float r = sqrtf(r2);
	cm->dis_[tid] += r2;
}

__global__ void InitProp(Prop* prop_, Box box);

void Particle::initProp(Box& box, bool isRandConfig, bool isInputFile)
{
	if (isInputFile) {
		char inputFilePath[100] = "./Input/mitotic_chromosome.txt";
		FILE* fp = fopen(inputFilePath, "r");
		for (int i = 0; i < sim::totalnumber; i++) {
			int temp1, temp2;
			fscanf(fp, "%d %d", &temp1, &temp2);
			fscanf(fp, "%f %f %f\n", &prop[i].pos[0], &prop[i].pos[1], &prop[i].pos[2]);
			fscanf(fp, "%d %d %d", &temp1, &temp2, &temp1);
			//printf("%f\n", prop[i].pos[0]);
			prop[i].IsActive = true;
			prop[i].periodic[0] = prop[i].periodic[1] = prop[i].periodic[2] = 0;
		}
		cudaMemcpy(prop_, prop, sizeof(Prop) * sim::totalnumber, cudaMemcpyHostToDevice);
	}
	else {
		if (!isRandConfig) {
			for (int i = 0; i < sim::totalnumber; i++) {
				if (i == 0) {
					for (int j = 0; j < DIMSIZE; j++) {
						prop[0].pos[j] = 0;
						prop[0].periodic[j] = 0;
					}
					prop[i].IsActive = true;
				}
				else {
					for (int j = 0; j < DIMSIZE; j++) {
						prop[i].pos[j] = prop[i - 1].pos[j] + 1.0 * lj::sigma / sqrtf(3);
						prop[i].periodic[j] = 0;
					}
					prop[i].IsActive = true;
				}
			}
			cudaMemcpy(prop_, prop, sizeof(Prop) * sim::totalnumber, cudaMemcpyHostToDevice);
		}
		else {
			float theta = var::u(var::e) * PI;
			float phi = var::u(var::e) * 2 * PI;

			for (int i = 0; i < sim::totalnumber; i++) {

				if (i < sim::activenumber) prop[i].IsActive = true;

				else prop[i].IsActive = false;

				if (i == 0) {
					for (int i = 0; i < DIMSIZE; i++) {
						prop[0].pos[i] = 0.5 * box.d[i];
						prop[0].periodic[i] = 0;
					}
				}
				else {
					theta = var::u(var::e) * PI;
					phi = var::u(var::e) * 2 * PI;

					float d[3];
					d[2] = rouse::b * cosf(theta); d[0] = rouse::b * sinf(theta) * cosf(phi); d[1] = rouse::b * sinf(theta) * sinf(phi);
					for (int j = 0; j < DIMSIZE; j++) {
						prop[i].pos[j] = prop[i - 1].pos[j] + d[j];
						prop[i].periodic[j] = 0;
						//mymodify(&prop[i].pos[j], &prop[i].periodic[j], box.d[j]);
					}
				}
			}
			cudaMemcpy(prop_, prop, sizeof(Prop) * sim::totalnumber, cudaMemcpyHostToDevice);
		}
	}
	for (int i = 0; i < sim::totalnumber; i++) {
		if (i < 0.5*sim::totalnumber) {
			prop[i].kappa = kick::kappa_short1;
		}
		else {
			prop[i].kappa = kick::kappa_short2;
		}
	}
	cudaMemcpy(prop_, prop, sizeof(Prop) * sim::totalnumber, cudaMemcpyHostToDevice);
}

__global__ void InitProp(Prop* prop_, Box box){
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid < sim::totalnumber_) {
		//init Isactive
		if (tid < sim::activenumber_) {
			prop_[tid].IsActive = true;
		}
		else {
			prop_[tid].IsActive = false;
		}

		//init position
		// simple cubic array
		int pos[3] = { 0,0,0 };
		int d = box.d[0];

		pos[2] = tid / (d * d);
		pos[0] = (tid - pos[2]) / d;
		pos[1] = tid % d;

		if (pos[0] % 2 == 1) {
			pos[1] = d-pos[1]-1;
		}

		for (int i = 0; i < DIMSIZE; i++) {
			prop_[tid].pos[i] = (float)pos[i];
		}
	}
}

//template <typename BOND>
//__global__ void CalcBonddU(Prop* prop_, KMC* kmc_, BOND fene, Box* box, curandState* states_);

/*
void Particle::updateKickPosition()
{
	cudaMemcpy(kmc.rate, kmc.rate_, sizeof(float) * sim::activenumber * sim::activenumber, cudaMemcpyDeviceToHost);
	cudaMemcpy(&kmc.sumRate, kmc.sumRate_, sizeof(float), cudaMemcpyDeviceToHost);

	float rd = var::u(var::e) * kmc.sumRate;
	//printf("%f\n", kmc.sumRate);
	float sum_rate = 0;
	int tid;
	bool succeed = false;
	for (int i = 0; i < sim::activenumber; i++) {
		for (int j = 0; j < sim::activenumber; j++) {
			if ( EXPRESSION ) {
				tid = i * sim::activenumber + j;
				sum_rate += kmc.rate[tid];
				kmc.rate[tid] = sum_rate;
				if (rd < sum_rate) {
					succeed = true;
					cudaMemcpy(prop_[i].pos, kmc.kickpos_[tid].e, sizeof(float) * DIMSIZE, cudaMemcpyDeviceToDevice);
					cudaMemcpy(prop_[j].pos, kmc.kickpos_[j*sim::activenumber + i].e, sizeof(float) * DIMSIZE, cudaMemcpyDeviceToDevice);
					break;
				}
			}
		}
		if (succeed) break;
	}
}
*/
__global__ void Update_flag(KMC* kmc_, Prop* prop_, float beta, curandState* state, float p10, float p_slide) {
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid < sim::activenumber_ * sim::activenumber_) {
		if (kmc_->flag_[tid] == 1) {
			float r = curand_uniform(state + tid);
			if (r < p10) {
				kmc_->flag_[tid] = 0;
			}
			else if (r < p_slide + p10) {
				int i = tid / sim::activenumber_;
				int j = tid % sim::activenumber_;
				//printf("%d %d\n", i, j);
				kmc_->flag_[tid] = 0;
				float U = 0.5 * kmc_->ks * ((prop_[i].pos[0] - prop_[j].pos[0]) * (prop_[i].pos[0] - prop_[j].pos[0]) +
					(prop_[i].pos[1] - prop_[j].pos[1]) * (prop_[i].pos[1] - prop_[j].pos[1]) +
					(prop_[i].pos[2] - prop_[j].pos[2]) * (prop_[i].pos[2] - prop_[j].pos[2]));
				int slide_id = tid;
				int i_, j_;
				if (i > j) {
					i_ = (i < sim::activenumber_ - 1) ? i + 1 : i;
					j_ = (j > 0) ? j - 1 : j;
				}
				slide_id = i_ * sim::activenumber_ + j_;
				float U_prime = 0.5 * kmc_->ks * ((prop_[i_].pos[0] - prop_[j_].pos[0]) * (prop_[i_].pos[0] - prop_[j_].pos[0]) +
					(prop_[i_].pos[1] - prop_[j_].pos[1]) * (prop_[i_].pos[1] - prop_[j_].pos[1]) +
					(prop_[i_].pos[2] - prop_[j_].pos[2]) * (prop_[i_].pos[2] - prop_[j_].pos[2]));
				float dU = U_prime - U;
				float rate = 1;
				if(dU > 0) rate = expf(-beta * dU);
				//printf("%f\n", rate);
				r = curand_uniform(state + tid);
				if (r < rate) { kmc_->flag_[slide_id] = 1; }
				else { kmc_->flag_[tid] = 1; }
				if(!(SHORTDIS_ || LONGDIS_)) { kmc_->flag_[slide_id] = 0; }
			}
		}
	}
}

__device__ int Poisson(float lambda, curandState* state, int tid);
__global__ void CalcGrapplingIndicator(KMC* kmc_, curandState* state, float tau, float* dis_ = nullptr, bool isCofRij = false) {
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid < sim::activenumber_ * sim::activenumber_) {

		float r = 0; float Cxy = 0; float rate = 0; int k = 0;

		if (isCofRij) r = sqrtf(dis_[tid]);

		r = sqrtf(dis_[tid]);

		int i = tid / sim::activenumber_;
		int j = tid % sim::activenumber_;
		if (kmc_->flag_[tid] == 1 && (i-j) <=4 ) {
			//printf("%d %d\n", i, j);
		}
		if (SHORTDIS || LONGDIS) {
			if (kmc_->flag_[tid] == 0) {
				int i = tid / sim::activenumber_;
				int j = tid % sim::activenumber_;
				if (isCofRij) {
					Cxy = 0.1 / r;
				}
				else {
					Cxy = 0.5 / fabsf(i - j);
				}
				if (SHORTDIS) {
					rate = Cxy * kmc_->ksb;
				}
				else {
					rate = Cxy * kmc_->klb;
				}
				k = Poisson(rate * tau, state, tid);
				if (k > 0) {
					kmc_->flag_[tid] = 1;
				}  
			}
		}
	}
}

__global__ void CalcRate(KMC* kmc_, Prop* prop_, curandState* state, float beta, float tau, bool isCofRij=false) {
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid < sim::activenumber_ * sim::activenumber_) {

		int i = tid / sim::activenumber_;
		int j = tid % sim::activenumber_;
		int k = 0; float Cxy = 0; float rate = 0; float dU = 0;

		if((SHORTDIS || LONGDIS)&& kmc_->flag_[tid] == 1) {
			/*
			{
				rate = kmc_->rate_[tid];
				//printf("%f %f\n", rate, kmc_->dU_[tid]);
				k = Poisson(rate * tau, state, tid);
			}
			*/
			//if (kmc_->flag_[tid] == 1 && (i - j == 1)) {
			//	printf("%d %d\n", i, j);
			//}
			{
				dU = kmc_->dU_[tid];
				if (dU < -1.0) dU = -1.0;
				if (SHORTDIS) {
					kmc_->rate_[tid] = 0.5 * (prop_[i].kappa + prop_[j].kappa) * expf(-kmc_->s * beta * dU);
				}
				else {
					kmc_->rate_[tid] = kmc_->kappa_long * expf(-kmc_->s * beta * dU);
				}
				rate = kmc_->rate_[tid];

				k = Poisson(rate * tau, state, tid);
			}
			atomicAdd(&prop_[i].kickNumber, k);
			atomicAdd(&prop_[j].kickNumber, k);

			for (int n = 0; n < DIMSIZE; n++) {
				//prop_[i].pos[n] += k * kmc_->kickpos_[tid].e[n];
				//prop_[j].pos[n] += kmc_->kickpos_[j * sim::activenumber_ + i].e[n];
				atomicAdd(&prop_[i].dposkmc[n], k * kmc_->kickdpos_[tid].e[n]);
				atomicAdd(&prop_[j].dposkmc[n], k * kmc_->kickdpos_[j * sim::activenumber_ + i].e[n]);

				//printf("____%f %f\n", prop_[i].pos[n], k * kmc_->kickdpos_[tid].e[n]);
			}
		}else {
			kmc_->rate_[tid] = 0;
		}
			
	}
}

__global__ void IsPosDivergency(Prop* prop_, int* isDivergencyed) {
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid < sim::totalnumber_) {

		float dis2 = 0;
		for (int i = 0; i < DIMSIZE; i++) {
			dis2 += prop_[tid].dposkmc[i] * prop_[tid].dposkmc[i];
		}
		float dis = sqrtf(dis2);
		/*
		if (dis < 0.5) {
			for (int i = 0; i < DIMSIZE; i++) {
				prop_[tid].pos[i] += prop_[tid].dposkmc[i];
			}
		}
		else {
			printf("dis:%f dis2:%f\n", dis, dis2);
		}
		*/
		if (dis > 0.05) {
			*isDivergencyed = 1;
		}
	}
}

__global__ void UpdateKickPosition(Prop* prop_) {
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid < sim::totalnumber_) {
		for (int i = 0; i < DIMSIZE; i++) {
			prop_[tid].pos[i] += prop_[tid].dposkmc[i];
		}
	}
}
/*
__global__ void CalcCumulateRate(KMC* kmc_, Prop* prop_) {
	int tid = threadIdx.x + blockDim.x * blockIdx.x;

	if (tid < sim::activenumber_ * sim::activenumber_) {

		bool succeed = false;

		if (tid == 0) {
			kmc_->key_[0] = -1;
			return;
		}

		while (!succeed) {
			if (atomicExch(&kmc_->key_[tid - 1], 0)) {

				kmc_->rate_[tid] = kmc_->rate_[tid] + kmc_->rate_[tid - 1];
				
				//printf("%d %f\n", tid, kmc_->rate_[tid]);

				kmc_->key_[tid] = -1;

				succeed = true;
			}
		}

		//if (tid == (sim::activenumber_ * sim::activenumber_ - 1)) {
		//	printf("nnnnn %f\n", kmc_->rate_[tid]);
		//}
		
		if (tid == 0) {
			if (randnumber < kmc_->rate_[tid]) {
				memcpy(prop_[i].pos, kmc_->kickpos_[tid].e, sizeof(float) * DIMSIZE);
				memcpy(prop_[j].pos, kmc_->kickpos_[j * sim::activenumber_ + i].e, sizeof(float) * DIMSIZE);
				*kmc_->pullstate_ = 1;
			}
			*kmc_->key_ = 0;
			succeed = true;
			*kmc_->isdone_ = 1;
		}

		while (!succeed && !*kmc_->pullstate_ && *kmc_->isdone_) {
			if (!atomicExch(kmc_->key_, 1)) {
				kmc_->rate_[tid] += kmc_->rate_[tid - 1];
				if (randnumber < kmc_->rate_[tid]) {
					memcpy(prop_[i].pos, kmc_->kickpos_[tid].e, sizeof(float) * DIMSIZE);
					memcpy(prop_[j].pos, kmc_->kickpos_[j * sim::activenumber_ + i].e, sizeof(float) * DIMSIZE);
					*kmc_->pullstate_ = 1;
				}
				*kmc_->key_ = 0;
				succeed = true;
			}
		}
	}
}
*/

__global__ void Curandinit(curandState* states_, unsigned int seed);

void Particle::curandinit()
{
	Curandinit<<<calc_blocks(sim::totalnumber * sim::totalnumber), THREADS_PERBLOCK >> >(states, seed);
}

__global__ void Curandinit(curandState* states_, unsigned int seed) {
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid >=sim::totalnumber_*sim::totalnumber_) return;

	curand_init(seed + tid, tid, 0, states_ + tid);
}

__global__ void Clear_force(Prop* prop_);

void Particle::clear_force()
{
	Clear_force << <calc_blocks(sim::totalnumber), THREADS_PERBLOCK >> > (prop_);
}

__global__ void Clear_force(Prop* prop_) {
	int tid = threadIdx.x + blockDim.x * blockIdx.x;

	if (tid >=sim::totalnumber_) { return; }

	for (int i = 0; i < DIMSIZE; i++) {
		prop_[tid].force[i] = 0;
		prop_[tid].dposkmc[i] = 0;
	}

	for (int i = 0; i < 2; i++) {
		prop_[tid].pe[i] = 0;
	}

	prop_[tid].kickNumber = 0;
}

__global__ void PerodicBoundary(Prop* prop_, Box* box);
void Particle::perodicBoundary(Box* box)
{
	PerodicBoundary << <calc_blocks(sim::totalnumber), THREADS_PERBLOCK >> > (prop_, box);
}

__global__ void PerodicBoundary(Prop* prop_, Box* box) {
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid >= sim::totalnumber_) { return; }

	for (int i = 0; i < DIMSIZE; i++) {
		mymodify(prop_[tid].pos + i, prop_[tid].periodic + i, box->d[i]);
	}
}

__global__ void UpdatePosition(Para para, Prop* prop_, curandState* states_);
void Particle::updatePosition()
{
	UpdatePosition << <calc_blocks(sim::totalnumber), THREADS_PERBLOCK >> > (para, prop_, states);
}

__global__ void UpdatePosition(Para para, Prop* prop_, curandState* states_) {
	int tid = threadIdx.x + blockDim.x * blockIdx.x;

	if (tid >= sim::totalnumber_) return;

	//printf("%f\n", para.dt);
	float dx = 0;

	for (int i = 0; i < DIMSIZE; i++) {
		dx = para.mu * prop_[tid].force[i] * para.dt + sqrtf(2 * para.mu/para.beta * para.dt) * curand_normal(states_ + tid);
		//dx = para.mu * prop_[tid].force[i] * para.dt;

		prop_[tid].pos[i] += dx;
	}
}

int Particle::get_bondnumber()
{
	return sim::totalnumber - 1;
}

/*
__global__ void Calc_dis_sq(Prop* prop_, Rg rg, int N);
void Particle::calcRg()
{
	cudaMemcpy(prop, prop_, sizeof(Prop) * sim::totalnumber, cudaMemcpyDepropiceToHost);
	float dist[3] = {0,0,0};
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < DIMSIZE; j++) {
			//printf("%f\n", prop[i].pos[j]);
			dist[j] += prop[i].pos[j];
			//printf("%f\n", dist[j]);
		}
	}
	for (int i = 0; i < DIMSIZE; i++) {
		rg.centerPos[i] = dist[i] / N;
		//printf("%f\n", rg.centerPos[i]);
	}
	
	cudaMemset(rg.Rg_, 0, sizeof(float));
	Calc_dis_sq << <calc_blocks(sim::totalnumber), THREADS_PERBLOCK >> > (prop_, rg, N);

	cudaMemcpy(&rg.Rg, rg.Rg_, sizeof(float), cudaMemcpyDepropiceToHost);
}
*/

/*
__global__ void Calc_dis_sq(Prop* prop_, Rg rg, int N) {
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid >=sim::totalnumber) return;
	float dist[DIMSIZE];
	for (int i = 0; i < DIMSIZE; i++) {
		dist[i] = prop_[tid].pos[i] - rg.centerPos[i];
	}
	rg.dis_sq_[tid] = calcDist_Squared(dist);
	//printf("%f\n", rg.dis_sq_[tid]);
	atomicAdd(rg.Rg_, rg.dis_sq_[tid] / N);
}
*/

__global__ void CalcBendingForce(Prop* prop_, float ka);
void Particle::calcBendingForce(float ka)
{
	CalcBendingForce << <calc_blocks(sim::totalnumber), THREADS_PERBLOCK >> > (prop_, ka);
}

__global__ void CalcBendingForce(Prop* prop_, float ka) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >=sim::totalnumber_ - 2) return;
	float posProp_12[3];
	float posProp_32[3];
	for (int i = 0; i < DIMSIZE; i++) {
		posProp_12[i] = prop_[tid].pos[i] - prop_[tid + 1].pos[i];
		posProp_32[i] = prop_[tid + 2].pos[i] - prop_[tid + 1].pos[i];
	}
	//calc Force

	float r_12_sq = calcDist_Squared(posProp_12);
	float r_32_sq = calcDist_Squared(posProp_32);

	float product_module = sqrtf(r_12_sq * r_32_sq) / ka;
	float dotProduct = posProp_12[0] * posProp_32[0] + posProp_12[1] * posProp_32[1] + posProp_12[2] * posProp_32[2];

	float force[3][3];
	for (int i = 0; i < 3; i++) {
		force[0][i] = (dotProduct * posProp_12[i] / r_12_sq - posProp_32[i]) / product_module;
		force[2][i] = (dotProduct * posProp_32[i] / r_32_sq - posProp_12[i]) / product_module;
		force[1][i] = -(force[0][i] + force[2][i]);
	}

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			atomicAdd(prop_[tid + j].force + i, force[j][i]);
		}
	}
}

__global__ void CalcSphereConfineForce(Particle p, Box* box, SPHRERCONFIE sphere);

void Particle::calcSphereConfineForce(Box* box, SPHRERCONFIE& sphere)
{
	CalcSphereConfineForce << <calc_blocks(sim::totalnumber), THREADS_PERBLOCK >> > (*this, box, sphere);
}

__global__ void CalcSphereConfineForce(Particle p, Box* box, SPHRERCONFIE sphere) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >=sim::totalnumber_) return;

	float dist[DIMSIZE];
	getUnreduceDistance(box->rc, p.prop_[tid].pos, dist);

	float r2 = 0;

	for (int i = 0; i < DIMSIZE; i++) {
		r2 += dist[i] * dist[i];
	}

	float r = sqrtf(r2);

	float ff = sphere.calcForce(r);

	for (int j = 0; j < DIMSIZE; j++) {
		p.prop_[tid].force[j] += ff * dist[j] / r;
	}
}

void Rg::memory()
{
	cudaMalloc((void**)&dis_sq_, sizeof(float) * sim::totalnumber);
	cudaMalloc((void**)&Rg_, sizeof(float));
}

void KMC::memory()
{
	rate = new float[sim::activenumber* sim::activenumber];

	cudaMalloc((void**)&dU_, sizeof(float) * sim::activenumber * sim::activenumber);

	cudaMalloc((void**)&rate_, sizeof(float) * sim::activenumber * sim::activenumber);

	cudaMalloc((void**)&kickpos_, sizeof(V3) * sim::activenumber * sim::activenumber);

	cudaMalloc((void**)&kickdpos_, sizeof(V3) * sim::activenumber * sim::activenumber);

	cudaMalloc((void**)&flag_, sizeof(int) * sim::activenumber * sim::activenumber);

	cudaMemset(flag_, 0, sizeof(int) * sim::activenumber * sim::activenumber);

	flag = new int[sim::activenumber * sim::activenumber];
}

void ContactMap::memory()
{
	cudaMalloc((void**)&CM_, sizeof(int) * sim::totalnumber * sim::totalnumber);
	cudaMemset(CM_, 0, sizeof(int) * sim::totalnumber * sim::totalnumber);
	CM = new int[sim::totalnumber * sim::totalnumber];

	cudaMalloc((void**)&dis_, sizeof(float) * sim::totalnumber * sim::totalnumber);
	cudaMemset(dis_, 0, sizeof(float) * sim::totalnumber * sim::totalnumber);
	dis = new float[sim::totalnumber * sim::totalnumber];
}

void ContactMap::deviceToHost()
{
	cudaMemcpy(CM, CM_, sizeof(int) * sim::totalnumber * sim::totalnumber, cudaMemcpyDeviceToHost);
	cudaMemcpy(dis, dis_, sizeof(float) * sim::totalnumber * sim::totalnumber, cudaMemcpyDeviceToHost);
}


__global__ void CalcAfterKickPosition(Prop* prop_, KMC* kmc_, Box* box) {

	int tid = threadIdx.x + blockDim.x * blockIdx.x;

	if (tid < sim::activenumber_ * sim::activenumber_) {
		int x = tid / sim::activenumber_;
		int y = tid % sim::activenumber_;

		if (x != y) {

			float dist[DIMSIZE];
			for (int i = 0; i < DIMSIZE; i++) {
				dist[i] = prop_[y].pos[i] - prop_[x].pos[i] + box->d[i] * (prop_[y].periodic[i] - prop_[x].periodic[i]);
			}

			float r2 = 0;
			for (int i = 0; i < DIMSIZE; i++) {
				r2 += dist[i] * dist[i];
			}

			for (int i = 0; i < DIMSIZE; i++) {
				kmc_->kickpos_[tid].e[i] = kmc_->l * dist[i] / sqrtf(r2) + prop_[x].pos[i];
				kmc_->kickdpos_[tid].e[i] = kmc_->l * dist[i] / sqrtf(r2);
				//printf("%f\n", kmc_->kickpos_[tid].e[i]);
			}
		}
	}
}

__device__ float calcCosine(float* pos1, float* pos2, float* pos3) {
	// Calculate bond vectors

	float3 vec1 = make_float3(pos2[0] - pos1[0], pos2[1] - pos1[1], pos2[2] - pos1[2]);
	float3 vec2 = make_float3(pos3[0] - pos2[0], pos3[1] - pos2[1], pos3[2] - pos2[2]);

	// Normalize bond vectors
	float dot11 = vec1.x* vec1.x + vec1.y*vec1.y + vec1.z*vec1.z;
	float dot22 = vec2.x*vec2.x + vec2.y*vec2.y + vec2.z*vec2.z;
	
	float dot12 = vec1.x * vec2.x + vec1.y * vec2.y + vec1.z * vec2.z;

	return (dot12 / sqrtf(dot11 * dot22));
}

__device__ float calcBendingPE(Prop* prop_, Kratky_Porod& angle, int i, float* posi) {

	float pe1 = 0; float costheta = 0;
	if (i - 1 > 0) {
		costheta = calcCosine(prop_[i - 2].pos, prop_[i - 1].pos, posi);
		pe1 = angle.calcPE(costheta);
	}

	float pe2 = 0;
	if (i != 0 && i != sim::activenumber_ - 1) {
		costheta = calcCosine(prop_[i - 1].pos, posi, prop_[i + 1].pos);
		pe2 = angle.calcPE(costheta);
	}

	float pe3 = 0;
	if (i + 1 < sim::activenumber_ - 1) {
		costheta = calcCosine(posi, prop_[i + 1].pos, prop_[i + 2].pos);
		pe3 = angle.calcPE(costheta);
	}

	return (pe1 + pe2 + pe3);
}

__device__ int Poisson(float lambda, curandState* state, int tid) {
	float L = expf(-lambda); int k = 0; float p = 1; float u = 0;
	int ret = 0;
	do {
		k += 1;
		u = curand_uniform(state + tid );
		//u = 0.1;
		p *= u;
	} while (p > L);
	ret = k - 1;
	//if (ret > 10) {
	//	printf("ret: %d\n", ret);
	//}
	return ret;
}

__global__ void CalcBendingdU(Prop* prop_, KMC* kmc_, Kratky_Porod angle) {
	int tid = threadIdx.x + blockDim.x * blockIdx.x;

	if (tid < sim::activenumber_ * sim::activenumber_) {
		int x = tid / sim::activenumber_;
		int y = tid % sim::activenumber_;

		if (abs(x-y)>0) {
			float ux = calcBendingPE(prop_, angle, x, prop_[x].pos);
			float uy = calcBendingPE(prop_, angle, y, prop_[y].pos);

			float ux_ = calcBendingPE(prop_, angle, x, kmc_->kickpos_[x*sim::activenumber_ + y].e);
			float uy_ = calcBendingPE(prop_, angle, y, kmc_->kickpos_[y*sim::activenumber_ + x].e);

			float du = (ux_ - ux) + (uy_ - uy);

			//printf("u %d %f %f %f %f\n", y, ux, uy, ux_, uy_);

			kmc_->dU_[tid] += du;

			//printf("____%f\n", kmc_->dU_[tid]);
		}
	}
}

