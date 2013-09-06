#ifndef DNN_NEURON_H
#define DNN_NEURON_H 1

#include <cstdio>
using namespace std;

class Layer;
//Layer::getInput(int,int,int,int,int,int);

class Neuron{
	public:
		int ix,iy,iz;		//The size of the inputs
		int lx,ly,lz;		//The location in the layer of this neuron
		//Neuron**** inputs;	//The neuron inputs to this layer
		float*** weights;	//The weights to all the inputs.
		float*** deltaWeights;	//The change in weights which get added when updateWeights is called
		float* bias;		//The threshold
		float* deltaBias;	//The change in threshold that gets set when updateWidths is called
		float errorYGradient;
		float errorZGradient;
		int lastBackPropPass;

		Neuron* getInput(int y, int x, int z);

		Layer* layer;
		
		virtual float getOutput(int pass)=0;

		Neuron();

		void addToYError(float toAdd, int pass);
};

#endif
