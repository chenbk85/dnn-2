#include "RegressionFunctions.h"
#include <cmath>
#include <cstdio>
using namespace std;

namespace RegressionFunctions{
	float squaredError(float expected, float calculated){
		float diff = expected-calculated;
		return diff*diff/(float)2.0;
	}

	float squaredErrorGradient(float expected, float calculated){
		return -(expected-calculated);
	}

	float softmaxError(float expected, float calculated){
		return -expected*log(calculated);
	}
	
	float softmaxErrorGradient(float expected, float calculated){
		return calculated-expected;
	}
}
