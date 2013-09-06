#ifndef DNN_STANDARD_UNIT_H
#define DNN_STANDARD_UNIT_H 1

#include "neuron.h"

class InputUnit:public Neuron{
	public:
		void setInputValue(float value);
		float getInputValue();
		float getOutput(int pass);
	private:
		float value;
};

#endif
