#include "activationFunctions.h"
#include <cmath>
using namespace std;

namespace ActivationFunctions{
	float relu(float x){
		return x>0?x:0;
	}

	float reluGradient(float x){
		return x<=0?0:1;
	}
	
	float logistic(float x){
		return 1/(1+exp(-x));
	}
	
	float logisticGradient(float x){
		float a = logistic(x);
		return a*(1-a);
	}

	float noGradient(float){return -1;}
}
