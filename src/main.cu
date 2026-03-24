
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#include "Particle.h"
#include "var.h"
#include "InputOutput.h"
#include "Interract.h"
#include "Cell.h"
#include "System.h"
#include <iostream>
#include <string>
#include "PullMethod.h"

void ParaProcess(int argc, char* argv[]);
void setTemperature(Particle& p, HARMONIC& bond, LJ& nobond1, SC& nobond, Kratky_Porod& angle);

#define KICK
#define WARM 1
int main(int argc, char* argv[]) {
	
	ParaProcess(argc, argv);
	cudaMemcpyToSymbol(sim::totalnumber_, &sim::totalnumber, sizeof(int));
	cudaMemcpyToSymbol(sim::activenumber_, &sim::activenumber, sizeof(int));

	Box box;
	Box* box_;
	cudaMalloc((void**)&box_, sizeof(Box));
	TOGPU(&box, box_);

	InputOutput inout;
	inout.outPara();

	//nobond
	//SC nobond = SC(lj::sigma, lj::epi, lj::Ecut, lj::ro, lj::cutoff);
	SC nobond = SC(lj::sigma, lj::epi, lj::Ecut, lj::ro, lj::cutoff, lj::rc, lj::rb, lj::barrier);

	//bond
	//FENE bond = FENE(fene::kb, fene::R0);
	HARMONIC bond = HARMONIC(rouse::k, 0);
	//LJ bond1 = LJ(lj::epi, lj::sigma, lj::cutoff);
	SC bond1 = SC(lj::sigma, lj::epi, lj::Ecut, lj::ro, lj::cutoff, lj::rc, lj::rb, lj::barrier);

	Kratky_Porod angle = Kratky_Porod(bend::ka);

	//SPHRERCONFIE sphere = SPHRERCONFIE(1.0, 0.5*box.d[0]);

	Cell c = Cell(1.12 * lj::cutoff, box);
	c.memory();
	c.initial();

	std::random_device seed;

	var::e.seed(var::seed);
	//std::cout << var::e() << std::endl;

	Particle p;
	p.init(box);

	PullMethod pull;
	pull.memory();

	System s = System();
	s.memory();
	cudaEvent_t start, stop;
	float elapsedTime = 0.0;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	//rouse relaxation 1000 s
	//sim::warm_cycles = 1e10;
	//inout.readAdj(p, "adj_153300.txt");
	inout.outTopo(p, box, true);
	cudaMemcpy(p.prop, p.prop_, sizeof(Prop) * sim::totalnumber, cudaMemcpyDeviceToHost);
	inout.appendXyzTrajectory(DIMSIZE, p, true);
	inout.appendKickFrequency(p, true);
	int cnt = 0;
	printf("warmstep is %d\n", sim::warm_cycles * sim::stepsPersecond);
#if WARM
	for (int i = 0; i < sim::warm_cycles * sim::stepsPersecond; i++) {
		//c.buildcelllist(p);
		s.runMD(p, c, box_, bond, bond1, nobond, pull);
		//if (i % (100*sim::stepsPersecond) == 0) {
		//	printf("it have been warmed %d s\n", 100 * cnt++);
		//	cudaMemcpy(p.prop, p.prop_, sizeof(Prop) * sim::totalnumber, cudaMemcpyDeviceToHost);
		//	inout.appendXyzTrajectory(DIMSIZE, p, false);
		//}
	}
	cudaMemcpy(p.prop, p.prop_, sizeof(Prop) * sim::totalnumber, cudaMemcpyDeviceToHost);
	inout.appendXyzTrajectory(DIMSIZE, p, false);
	printf("rouse chain relaxation is done!\n");

#endif 

	int cycles = 0;
	float t = 0;
	int MD_CYCLE;

	float dt = sim::dt;
	int lastTotalKicktime = 0;

#ifdef KICK
	int countkick = 0;
	int count = 0;
	//kick ~1e5s
	s.tau = 1e3 *dt;
	while (p.kmc->totalKmcTime < sim::kickTime) {

		//cudaMemcpy(p.prop, p.prop_, sizeof(Prop) * sim::totalnumber, cudaMemcpyDeviceToHost);
		//inout.appendXyzTrajectory(DIMSIZE, p, false);

		//cudaMemcpy(p.prop, p.prop_, sizeof(Prop) * sim::totalnumber, cudaMemcpyDeviceToHost);
		//inout.appendXyzTrajectory(DIMSIZE, p, false);


		//MD_CYCLE = s.tau/dt;
		MD_CYCLE = 100;
		for (int i = 0; i < MD_CYCLE; i++) {
			s.runMD(p, c, box_, bond, bond1, nobond, pull);
		}

		s.runKMC(p, c, bond, bond1, nobond, angle, box_);

		countkick++;

		if ( countkick % 100 == 0) {
			cudaMemcpy(p.prop, p.prop_, sizeof(Prop) * sim::totalnumber, cudaMemcpyDeviceToHost);
			inout.appendXyzTrajectory(DIMSIZE, p, false);
			inout.appendKickFrequency(p, false);
			inout.outAdj(p, countkick);
			printf("have done %f s!\n", p.kmc->totalKmcTime);
		}

		p.kmc->totalKmcTime += s.tau;

		/*
		if (1 / media::beta < 0.2) {
			if (cnt % 200 == 0) {
				media::beta = 1.0 / (1.0/media::beta - 0.1);

				setTemperature(p, bond, bond1, nobond, angle);

				printf("beta: %f\n", 1 / media::beta);
			}
		}
		*/

			//inout.appendCorrelationTrajectory(p, false);
			/*
			if (cnt % 100 == 0) {
				inout.appendXyzTrajectory(DIMSIZE, p, false);
				//inout.appendCenterPosition(p, false);
				//inout.outDistanceMap(p, count);
				printf("have done %d s and totoalKicktime is %f!\n", cnt, p.kmc.totalKmcTime);
			}
			*/
	}
#else
	int count = 0;
	int countstep = 0;
	int countct = 0;
	for (int i = 0; i < sim::kickTime; i++) {
		for (int j = 0; j < sim::stepsPersecond; j++) {
			p.runMD(box_, bond, nobond);
			countstep++;
		}
		cnt++;

		p.distanceMap(box_);

		count++;

		cudaMemcpy(p.prop, p.prop_, sizeof(Prop) * sim::totalnumber, cudaMemcpyDeviceToHost);
		inout.appendCorrelationTrajectory(p, false);

		if (cnt % 100 == 0) {
			inout.appendXyzTrajectory(DIMSIZE, p, false);
			inout.appendCenterPosition(p, false);
			inout.outDistanceMap(p, count);
			printf("have done %d s and totoalKicktime is %f!\n", cnt, p.kmc.totalKmcTime);
		}
		
	}
#endif // DEBUG
}

void ParaProcess(int argc, char* argv[]) {
	for (int i = 1; i < argc; ++i) {
		std::string arg = argv[i];
		if ((arg == "-kappa" || arg == "--kappa") && i + 1 < argc) {
			kick::kappa_short1 = std::stof(argv[++i]); // 增加i以跳过参数的值
		}
		else if ((arg == "-l" || arg == "--l") && i + 1 < argc) {
			kick::l = std::stof(argv[++i]);
		}
		else if ((arg == "-s" || arg == "--s") && i + 1 < argc) {
			kick::s0 = std::stof(argv[++i]);
		}
		else if ((arg == "-tn" || arg == "--tn") && i + 1 < argc) {
			sim::totalnumber = std::stoi(argv[++i]);
		}
		else if ((arg == "-an" || arg == "--an") && i + 1 < argc) {
			sim::activenumber = std::stoi(argv[++i]);
		}
		else if ((arg == "-ensemble" || arg == "--ensemble") && i + 1 < argc) {
			sim::ensembleID = std::stoi(argv[++i]);
		}
		else if ((arg == "-seed" || arg == "--seed") && i + 1 < argc) {
			var::seed = std::stoi(argv[++i]);
		}
		else {
			std::cerr << "unkonwing para" << arg << std::endl;
		}
	}
}

/*
void setTemperature(Particle& p, FENE& bond, LJ& bond1, SC& nobond, Kratky_Porod& angle) {
	p.para.beta = media::beta;
	bond.kb = 30 / (media::beta * lj::sigma * lj::sigma);
	bond1.epsion = 1.0/(media::beta);
	nobond.epsion = 1.0 / (media::beta);
	nobond.Ecut = 4 * nobond.epsion;
	angle.ka = 2 / (media::beta);
}
*/

void setTemperature(Particle& p, HARMONIC& bond, LJ& bond1, SC& nobond, Kratky_Porod& angle) {
	p.para.beta = media::beta;
	bond.k = 3 / (media::beta * rouse::b * rouse::b);
	bond1.epsion = 1.0 / (media::beta);
	nobond.epsion = 1.0 / (media::beta);
	nobond.Ecut = 4 * nobond.epsion;
	angle.ka = 2 / (media::beta);
}
