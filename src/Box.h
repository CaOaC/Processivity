#pragma once
#include "Macro.h"
#include "var.h"

class Box
{
public:
	float d[DIMSIZE] = {var::boxlength, var::boxlength, var::boxlength};
	float rc[DIMSIZE] = { 0.5f* var::boxlength, 0.5f * var::boxlength, 0.5f * var::boxlength };

public:
	Box();
	~Box();
};


