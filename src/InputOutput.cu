#include "InputOutput.h"
#include <string>

InputOutput::InputOutput()
{
}

void InputOutput::outPara()
{
    FILE* pFile;

    char FPATH[100] = "";
    sprintf(FPATH, "./Output/para_kappa%.1f_%.1f.txt", kick::kappa_short1, kick::kappa_long);

    pFile = fopen(FPATH, "w");

    fprintf(pFile, "ACTIVENUMBER: %d\n", sim::activenumber);
    fprintf(pFile, "PASSIVENUMBER: %d\n", sim::totalnumber - sim::activenumber);
    fprintf(pFile, "kappa: %f\n", kick::kappa_short1);
    fprintf(pFile, "kappa_long:%f\n", kick::kappa_long);
    fprintf(pFile, "l: %f\n", kick::l);
    fprintf(pFile, "s0: %f\n", kick::s0);
    fprintf(pFile, "beta: %f\n", media::beta);
    fprintf(pFile, "mu: %f\n", media::mu);
    fprintf(pFile, "boxLength: %f\n", var::boxlength);
    fprintf(pFile, "totalCycles: %d\n", sim::kickTime);

    fclose(pFile);

}

void InputOutput::appendXyzTrajectory(unsigned int dimension, const Particle& particles, bool clearFile)
{
    FILE* pFile;

    char FPATH[100] = "";
    sprintf(FPATH, "./Output/trajectory_kappa%.1f_%.1f_%.2f_%.2f_%d.dump", kick::kappa_short1, kick::kappa_long, kick::p10, kick::p_slide, sim::ensembleID);
    // Wipe existing trajectory file.
    if (clearFile)
    {
        pFile = fopen(FPATH, "w");
        fclose(pFile);
        return;
    }

    pFile = fopen(FPATH, "a");
    fprintf(pFile, "ITEM: TIMESTEP\n%f\n", particles.kmc->totalKmcTime);
    fprintf(pFile, "ITEM: BOX BOUNDS pp pp pp\n0 %f\n0 %f\n0 %f\n", var::boxlength, var::boxlength, var::boxlength);
    fprintf(pFile, "ITEM: NUMBER OF ATOMS\n%d\n", sim::totalnumber);
    fprintf(pFile, "ITEM: ATOMS id type x y z mycolor ix iy iz\n");
    for (unsigned int i = 0; i < sim::totalnumber; i++)
    {
        if (i < sim::activenumber) {
            if (particles.prop[i].IsActive) {
                fprintf(pFile, "%d 1 %5.4f %5.4f %5.4f %d %d %d %d\n",
                    i + 1, particles.prop[i].pos[0], particles.prop[i].pos[1], (dimension == 3) ? particles.prop[i].pos[2] : 0, particles.prop[i].kickNumber, particles.prop[i].periodic[0], particles.prop[i].periodic[1], particles.prop[i].periodic[2]);
            }
            else {
                fprintf(pFile, "%d 1 %5.4f %5.4f %5.4f %d %d %d %d\n",
                    i + 1, particles.prop[i].pos[0], particles.prop[i].pos[1], (dimension == 3) ? particles.prop[i].pos[2] : 0, particles.prop[i].kickNumber, particles.prop[i].periodic[0], particles.prop[i].periodic[1], particles.prop[i].periodic[2]);
            }
        }
        else {
            fprintf(pFile, "%d 2 %5.4f %5.4f %5.4f %d %d %d %d\n",
                i + 1, particles.prop[i].pos[0], particles.prop[i].pos[1], (dimension == 3) ? particles.prop[i].pos[2] : 0, particles.prop[i].kickNumber, particles.prop[i].periodic[0], particles.prop[i].periodic[1], particles.prop[i].periodic[2]);
        }

    }
    fclose(pFile);
}

void InputOutput::appendKickFrequency(Particle& particles, bool clearFile)
{
    FILE* fp;
    // wipe existing trajectory file
    char FPATH[100] = "";
    sprintf(FPATH, "./Output/kicknumber.txt");
    if (clearFile) {
        fp = fopen(FPATH, "w");
        fclose(fp);
    }
    else {
        fp = fopen(FPATH, "a");
        for (unsigned int i = 0; i < sim::totalnumber; i++) {
            fprintf(fp, "%d ", particles.prop[i].kickNumber);
        }
        fprintf(fp, "\n");
        fclose(fp);
    }
}

void InputOutput::appendCenterPosition(const Particle& particles, bool clearFile)
{
    FILE* fp;
    // Wipe existing trajectory file.
    char FPATH[100] = "";
    sprintf(FPATH, "./Output/CenterPOS_ems_%d.txt", sim::ensembleID);
    if (clearFile)
    {
        fp = fopen(FPATH, "w");
        fclose(fp);
    }
    else {
        fp = fopen(FPATH, "a");

        float x = 0; float y = 0; float z = 0;

        for (int i = 0; i < sim::totalnumber; i++) {
            x += (particles.prop[i].pos[0] + particles.prop[i].periodic[0] * var::boxlength);
            y += (particles.prop[i].pos[1] + particles.prop[i].periodic[1] * var::boxlength);
            z += (particles.prop[i].pos[2] + particles.prop[i].periodic[2] * var::boxlength);
        }

        fprintf(fp, "%f %f %f\n", x / sim::totalnumber, y / sim::totalnumber, z / sim::totalnumber);

        fclose(fp);
    }
}

void InputOutput::outContactMap(Particle& p)
{
    p.cm->deviceToHost();
    FILE* pFile;
    char FPATH[100] = "";
    sprintf(FPATH, "./Output/contactmap_ems_%d.txt", sim::ensembleID);
    pFile = fopen(FPATH, "w");
    for (int i = 0; i < sim::totalnumber; i++) {
        for (int j = 0; j < sim::totalnumber; j++) {
            fprintf(pFile, "%d ", p.cm->CM[i * sim::totalnumber + j]);
        }
        fprintf(pFile, "\n");
    }
    fclose(pFile);
}

void InputOutput::outDistanceMap(Particle& p, int cnt)
{
    p.cm->deviceToHost();
    FILE* pFile;
    char FPATH[100] = "";
    sprintf(FPATH, "./Output/distancemap_ems_%d.txt", sim::ensembleID);
    pFile = fopen(FPATH, "w");
    for (int i = 0; i < sim::totalnumber; i++) {
        for (int j = 0; j < sim::totalnumber; j++) {
            fprintf(pFile, "%f ", p.cm->dis[i * sim::totalnumber + j]/cnt);
        }
        fprintf(pFile, "\n");
    }
    fclose(pFile);
}

void InputOutput::appendCorrelationTrajectory(const Particle& particles, bool clearFile)
{
    FILE* fp;
    //Wipe existing trajectory file.
    char FPATH[100] = "";
    sprintf(FPATH, "./Output/trajForCorrelation_ems_%d.txt", sim::ensembleID);

    if (clearFile) {
        fp = fopen(FPATH, "w");
        fclose(fp);
        return;
    }

    fp = fopen(FPATH, "a");

    float x = 0; float y = 0; float z = 0;

    for (int i = 0; i < sim::totalnumber; i++) {
        x = particles.prop[i].pos[0] + particles.prop[i].periodic[0] * var::boxlength;
        y = particles.prop[i].pos[1] + particles.prop[i].periodic[1] * var::boxlength;
        z = particles.prop[i].pos[2] + particles.prop[i].periodic[2] * var::boxlength;
        fprintf(fp, "%f %f %f\n", x, y, z);
    }
    fclose(fp);
}

void InputOutput::readTopo(Particle& p)
{
    FILE* fp;
    char FPATH[100] = "";
    sprintf(FPATH, "./Input/input.txt");

    fp = fopen(FPATH, "r");

    for (int i = 0; i < sim::totalnumber; i++) {
        //read atomid atomtype
        int atominfo[2] = {0, 0};
        for (int j = 0; j < 2; j++) {
            fscanf(fp, "%d", &atominfo[j]);
        }
        //read atom position
        for (int j = 0; j < DIMSIZE; j++) {
            fscanf(fp, "%f", &p.prop[i].pos[j]);
        }
        //read periodic number
        for (int j = 0; j < DIMSIZE; j++) {
            fscanf(fp, "%d", &p.prop[i].periodic[j]);
        }
    }
    fclose(fp);
    cudaMemcpy(p.prop_, p.prop, sizeof(Prop) * sim::totalnumber, cudaMemcpyHostToDevice);
}

void InputOutput::outTopo(Particle& p, Box& box, bool clearFile)
{
    FILE* pFile;

    char FPATH[100] = "";
    sprintf(FPATH, "./Output/TOPO_kappa_%.1f_%.1f.data", kick::kappa_short1, kick::kappa_long);

    //Wipe existing trajectory file.
    if (clearFile) {
        pFile = fopen(FPATH, "w");
        fclose(pFile);
    }

    int total_number = p.get_bondnumber();

    pFile = fopen(FPATH, "a");
    fprintf(pFile, "LAMMPS data file for Rousechain generated by c++\n");
    fprintf(pFile, "\n\n");
    fprintf(pFile, "%d atoms\n", sim::totalnumber);
    fprintf(pFile, "3 atom types\n");
    fprintf(pFile, "1 bond types\n");
    fprintf(pFile, "%d bonds\n", total_number);
    fprintf(pFile, "\n\n");
    fprintf(pFile, "0 %f xlo xhi\n", box.d[0]);
    fprintf(pFile, "0 %f ylo yhi\n", box.d[1]);
    fprintf(pFile, "0 %f zlo zhi\n", box.d[2]);
    fprintf(pFile, "\n\n");
    fprintf(pFile, "Masses\n");
    fprintf(pFile, "\n");
    fprintf(pFile, "1 1.0\n");
    fprintf(pFile, "2 1.0\n");
    fprintf(pFile, "3 1.0\n");
    fprintf(pFile, "\n");
    fprintf(pFile, "Atoms\n\n");

    for (int i = 0; i < sim::totalnumber; i++) {
        if (i < sim::activenumber) {
            if (p.prop[i].IsActive) {
                fprintf(pFile, "%d %d %f %f %f\n", i + 1, 1, p.prop[i].pos[0], p.prop[i].pos[1], p.prop[i].pos[2]);
            }
            else {
                fprintf(pFile, "%d %d %f %f %f\n", i + 1, 1, p.prop[i].pos[0], p.prop[i].pos[1], p.prop[i].pos[2]);
            }
        }
        else
            fprintf(pFile, "%d %d %f %f %f\n", i + 1, 2, p.prop[i].pos[0], p.prop[i].pos[1], p.prop[i].pos[2]);
    }
    fprintf(pFile, "\n\nBonds\n\n");
    int cnt = 0;
    for (int i = 0; i < sim::totalnumber-1; i++) {   
        bool output = true;
        if (output) {
            fprintf(pFile, "%d %d %d %d\n", ++cnt, 1, i + 1, i+2);
        }
    }
    fclose(pFile);
}

void InputOutput::outAdj(Particle& p, int t)
{
    cudaMemcpy(p.kmc->flag, p.kmc->flag_, sizeof(int) * sim::activenumber * sim::activenumber, cudaMemcpyDeviceToHost);
    FILE* fp;
    char FPATH[100] = "";
    sprintf(FPATH, "./Output/adj/adj_%d.txt",t);
    fp = fopen(FPATH, "w");
    for (int i = 0; i < sim::activenumber * sim::activenumber; i++) {
        fprintf(fp, "%d\n", p.kmc->flag[i]);
    }
    fclose(fp);
}

void InputOutput::readAdj(Particle& p, const char* filename) {
    FILE* fp;
    char FPATH[100] = "";
    sprintf(FPATH, "./Input/%s", filename);
    fp = fopen(FPATH, "r");
    for (int i = 0; i < sim::activenumber * sim::activenumber; i++) {
        fscanf(fp, "%d", &p.kmc->flag[i]);
    }
    fclose(fp);
    cudaMemcpy(p.kmc->flag_, p.kmc->flag, sizeof(int) * sim::activenumber * sim::activenumber, cudaMemcpyHostToDevice);
}
