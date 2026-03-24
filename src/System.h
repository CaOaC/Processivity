#pragma once
#include "Particle.h"
#include "Cell.h"
#include "template.h"
#include "PullMethod.h"

class System
{
public:
	System();
	~System();
	float tau;
	int* isDivergencyed;
	void memory();
	template <typename BOND, typename BOND1, typename NOBOND>
	void runMD(Particle&  p, Cell& c, Box* box, BOND& bond, BOND1& bond1, NOBOND& nobond, PullMethod& pull);
	template<typename BOND, typename BOND1, typename NOBOND>
	void runKMC(Particle& p, Cell& c, BOND& bond, BOND1& bond1, NOBOND& nobond, Kratky_Porod& angle, Box* box);
};
__global__ void CalcDistance(Prop* prop_, float* dis_);

template <typename BOND, typename BOND1, typename NOBOND>
inline void System::runMD(Particle& p, Cell& c, Box* box, BOND& bond, BOND1& bond1, NOBOND& nobond, PullMethod& pull)
{
	p.clear_force();

	//no_bond_force(box, nobond);

	c.buildcelllist(p);

	c.calcNoBondForce(p, nobond, box);

	p.bond_force(box, bond, bond1);

	p.calcBendingForce(bend::ka);

	//p.calcEffectiveForce()
	pull.calcDirection(p.prop_);

	pull.addPullForce(p.prop_);

	p.updatePosition();

	p.perodicBoundary(box);
}

template<typename BOND, typename BOND1, typename NOBOND>
inline void System::runKMC(Particle& p, Cell& c, BOND& bond, BOND1& bond1, NOBOND& nobond, Kratky_Porod& angle, Box* box)
{
	c.buildcelllist(p);

	CalcNoBondU1 << <calc_blocks(sim::activenumber * sim::activenumber), THREADS_PERBLOCK >> > (p.prop_, p.kmc_, nobond, box, p.states);

	if (sim::isCofRij) {
		CalcDistance << <calc_blocks(sim::activenumber * sim::activenumber), THREADS_PERBLOCK >> > (p.prop_, p.dis_);
	}

	CalcAfterKickPosition << <calc_blocks(sim::activenumber * sim::activenumber), THREADS_PERBLOCK >> > (p.prop_, p.kmc_, box);

	CalcBonddU << <calc_blocks(sim::activenumber * sim::activenumber), THREADS_PERBLOCK >> > (p.prop_, p.kmc_, bond, bond1, box, p.states);

	CalcNoBondU2 << < calc_blocks(sim::activenumber * sim::activenumber), THREADS_PERBLOCK >> > (c, p.prop_, p.kmc_, nobond, box, p.states);

	CalcBendingdU << < calc_blocks(sim::activenumber * sim::activenumber), THREADS_PERBLOCK >> > (p.prop_, p.kmc_, angle);

	CalcGrapplingIndicator << <calc_blocks(sim::activenumber * sim::activenumber), THREADS_PERBLOCK >> > (p.kmc_, p.states, this->tau, p.dis_, sim::isCofRij);

	CalcRate << <calc_blocks(sim::activenumber * sim::activenumber), THREADS_PERBLOCK >> > (p.kmc_, p.prop_, p.states, media::beta, this->tau, sim::isCofRij);

	Update_flag << <calc_blocks(sim::activenumber * sim::activenumber), THREADS_PERBLOCK >> > (p.kmc_, p.prop_, media::beta, p.states, kick::p10, kick::p_slide);
	/*
	cudaDeviceSynchronize();
	* isDivergencyed = 0;
	//printf("%d\n", *isDivergencyed);

	IsPosDivergency << <calc_blocks(sim::totalnumber), THREADS_PERBLOCK >> > (p.prop_, isDivergencyed);
	cudaDeviceSynchronize();

	if (!(*isDivergencyed)) {
		UpdateKickPosition << <calc_blocks(sim::totalnumber), THREADS_PERBLOCK >> > (p.prop_);
	}
	else {
		printf("is divergencyed!\n");
	}
	*/
	UpdateKickPosition << <calc_blocks(sim::totalnumber), THREADS_PERBLOCK >> > (p.prop_);

	p.perodicBoundary(box);

}
