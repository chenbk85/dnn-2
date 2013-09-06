#ifndef DNN_STANDARD_UNIT_H
#define DNN_STANDARD_UNIT_H 1

#include "neuron.h"

class StandardUnit : public Neuron{
	public:
		StandardUnit();
		float getOutput(int pass);
	private:
		int passCount;
		float outputComputed;
		float momentum;
};

#endif
