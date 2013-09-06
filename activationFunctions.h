#ifndef DNN_ACTIVATION_FUNCTIONS_H
#define DNN_ACTIVATION_FUNCTIONS_H 1

namespace ActivationFunctions{
	float relu(float);
	float reluGradient(float);
	float noGradient(float);
	
	float logistic(float);
	float logisticGradient(float);
}

#endif
