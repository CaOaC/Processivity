#include "Cell.h"
#include <math.h>
#include "curand_kernel.h"
#include <stdio.h>

__global__ void Buildcelllist(Cell c, Prop* prop_);

Cell::Cell()
{
}

Cell::~Cell()
{
}

Cell::Cell(float cutoff, Box& box)
{
	for (int i = 0; i < DIMSIZE; i++) {
		int num = box.d[i] / cutoff;
		dim[i] = (num > 0) ? num : 1;
	}
	cell_length = cutoff;
}

void Cell::buildcelllist(Particle& p)
{
	reset_celllist();
	Buildcelllist << <calc_blocks(sim::totalnumber), THREADS_PERBLOCK>> > (*this, p.prop_);
}

__device__ __host__ int Cell::getcellnum()
{
	int ret = 1;
	for (int i = 0; i < DIMSIZE; i++) {
		ret *= dim[i];
	}
	return ret;
}

void Cell::memory()
{
	cudaMalloc((void**)&cell_list, sizeof(Node) * sim::totalnumber);
	cudaMalloc((void**)&cell_head, sizeof(int) * getcellnum());
	cudaMemset(cell_head, -1, sizeof(int) * getcellnum());
	cudaMalloc((void**)&key, sizeof(int) * getcellnum());
	cudaMemset(key, -1, sizeof(int) * getcellnum());
}


void Cell::initial()
{
}

void Cell::reset_celllist()
{
	cudaMemset(cell_head, -1, sizeof(int) * getcellnum());
}


__global__ void Buildcelllist(Cell c, Prop* prop_) {
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid >= sim::totalnumber_) return;
	int cellid[DIMSIZE];
	for (int i = 0; i < DIMSIZE; i++) {
		cellid[i] = prop_[tid].pos[i]/c.cell_length;
		check_cell_pos_overflow(cellid + i, c.dim[i]);
		c.cell_list[tid].cellid[i] = cellid[i];
	}
	
	int cell_id = (DIMSIZE > 2) ? cellid[0] * c.dim[1] + cellid[1] + cellid[2] * c.dim[0] * c.dim[1] : cellid[0] * c.dim[1] + cellid[1];

	//printf("%d\n", c.cell_list[tid].cellid);
	bool succed = false;
	while(!succed){
		if (atomicExch(c.key + cell_id, 0))
		{
			c.cell_list[tid].next = c.cell_head[cell_id];
			c.cell_head[cell_id] = tid;
			__threadfence();

			c.key[cell_id] = -1;
			succed = true;
		}
	}
	//printf("%d\n", c.cell_list[tid].next);
}

__global__ void CalcNoBondForce(Prop* prop_, Cell c, SC nobond1, Box* box);
void Cell::calcNoBondForce(Particle& p, SC& nobond, Box* box)
{
	CalcNoBondForce << <calc_blocks(27 * sim::totalnumber), THREADS_PERBLOCK >> > (p.prop_, *this, nobond, box);
}

__global__ void CalcNoBondForce(Prop* prop_, Cell c, SC nobond, Box* box) {
	int offset = threadIdx.x + blockIdx.x * blockDim.x;

	int tid = int(offset / ((DIMSIZE > 2) ? 27 : 9));
	int id_in_27cells = offset % ((DIMSIZE > 2) ? 27 : 9);

	if (tid >= sim::totalnumber_) return;

	int cellid[DIMSIZE];
	cellid[2] = id_in_27cells / DIMSIZE / DIMSIZE;
	cellid[0] = id_in_27cells % (DIMSIZE * DIMSIZE) / DIMSIZE;
	cellid[1] = id_in_27cells % (DIMSIZE * DIMSIZE) % DIMSIZE;

	//printf("%d %d %d\n", cellid[0], cellid[1], cellid[2]);
	for (int i = 0; i < DIMSIZE; i++) {
		cellid[i] += (c.cell_list[tid].cellid[i] - 1);
		//if (cellid[i] < 0 || cellid[i] > c.dim[i]) return;
		check_cell_pos_overflow(cellid + i, c.dim[i]);
		//printf("%d %d %d\n", cellid[0], cellid[1], cellid[2]);
	}

	int cell_id = (DIMSIZE > 2) ? cellid[0] * c.dim[1] + cellid[1] + cellid[2] * c.dim[0] * c.dim[1] : cellid[0] * c.dim[1] + cellid[1];

	//printf("%d %d %d %d\n", cell_id, c.dim[0], c.dim[1], c.dim[2]);
	int tid_compare = c.cell_head[cell_id];
	float dist[DIMSIZE];
	while (tid_compare != -1) {
		if (tid_compare > sim::totalnumber_) printf("wrong %d\n", tid_compare);
		if (abs(tid-tid_compare)<2) {
			//printf("dadadad\n");
			tid_compare = c.cell_list[tid_compare].next;
			continue;
		}
		else {
			get_reduced_distance(prop_[tid_compare].pos, prop_[tid].pos, dist, box);
			float r2 = 0;
			for (int i = 0; i < DIMSIZE; i++) {
				r2 += dist[i] * dist[i];
			}
			//printf("%f\n", r2);
			float r = sqrtf(r2);
			//if (r < 0.8) r = 0.8;
			float ff = nobond.calcForce(r);
			//if(ff > 200) printf("%f\n", ff);
			for (int i = 0; i < DIMSIZE; i++) {
				//printf("%f\n", ff * dist[i] / sqrtf(r2));
				atomicAdd(&prop_[tid].force[i], ff * dist[i] / sqrtf(r2));
			}
			tid_compare = c.cell_list[tid_compare].next;
		}
	}
}

__global__ void CountCij(Cell c, Particle p, Box* box);

void Cell::countCij(Particle& p, Box* box)
{
	CountCij << <calc_blocks(27 * sim::totalnumber), THREADS_PERBLOCK >> > (*this, p, box);
}

__global__ void CountCij(Cell c, Particle p, Box* box) {
	int offset = threadIdx.x + blockIdx.x * blockDim.x;
	int tid = int(offset /  27 );
	int id_in_27cells = offset % (27);

	if (tid >= sim::totalnumber_) return;

	int cellid[DIMSIZE];
	cellid[2] = id_in_27cells / DIMSIZE / DIMSIZE;
	cellid[0] = id_in_27cells % (DIMSIZE * DIMSIZE) / DIMSIZE;
	cellid[1] = id_in_27cells - cellid[2] * DIMSIZE * DIMSIZE - cellid[0] * DIMSIZE;

	for (int i = 0; i < DIMSIZE; i++) {
		cellid[i] += (c.cell_list[tid].cellid[i] - 1);
		//if (cellid[i] < 0 || cellid[i] > c.dim[i]) return;
		check_cell_pos_overflow(cellid + i, c.dim[i]);
		//printf("%d %d %d\n", cellid[0], cellid[1], cellid[2]);
	}

	int cell_id = cellid[0] * c.dim[1] + cellid[1] + cellid[2] * c.dim[0] * c.dim[1];

	int tid_compare = c.cell_head[cell_id];
	float dist[DIMSIZE];
	//float unreduceDist[DIMSIZE];

	while (tid_compare != -1) {

		get_reduced_distance(p.prop_[tid_compare].pos, p.prop_[tid].pos, dist, box);
		//getUnreduceDistance(p.prop_[tid_compare].pos, p.prop_[tid].pos, unreduceDist);

		float r2 = 0;
		float r2_ = 0;
		for (int i = 0; i < DIMSIZE; i++) {
			r2 += dist[i] * dist[i];
		}
		//printf("%f\n", r2);
		float r = sqrtf(r2);
		if (r < 2) {
			atomicAdd(&p.cm->CM_[tid * sim::totalnumber_ + tid_compare], 1);
		}
		tid_compare = c.cell_list[tid_compare].next;
	}
}