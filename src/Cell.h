#pragma once
#include "Macro.h"
#include "Box.h"
#include "device_launch_parameters.h"
#include "Particle.h"

struct Node {
	int next = -1;
	int cellid[DIMSIZE];
};

class Cell
{
public:
	unsigned int dim[DIMSIZE];
	float cell_length;
	float cell_num;

public:
	Node* cell_list;
	int* cell_head;
	int* key;

public:
	__host__ __device__ Cell();
	__host__ __device__ ~Cell();
	Cell(float cutoff, Box&);
	void reset_celllist();
	void buildcelllist(Particle&);
	void countCij(Particle&, Box* box);
	__device__ __host__ int getcellnum();
	void memory();
	void calcNoBondForce(Particle& p, SC& nobond1, Box* box);
	void initial();
};
