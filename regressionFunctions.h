#ifndef DNN_REGRESSION_FUNCTIONS_H
#define DNN_REGRESSION_FUNCTIONS_H 1

namespace RegressionFunctions{
	float squaredError(float expected, float calculated);

	float squaredErrorGradient(float expected, float calculated);
	
	float softmaxError(float expected, float calculated);
	
	float softmaxErrorGradient(float expected, float calculated);
}

#endif
