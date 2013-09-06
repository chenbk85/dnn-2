#include "InputLayer.h"

#include "InputUnit.h"
#include "activationFunctions.h"

#include <algorithm>
#include <cassert>
#include <cstdio>

using namespace std;

Neuron**** InputLayer::initNeuronArray(int width, int height, int channels){
	Neuron**** array = new Neuron*** [height];
	for(int i =0; i<height;i++){
		array[i] = new Neuron** [width];
		for(int j = 0; j < width; j++){
			array[i][j] = new Neuron*[channels];
			for(int k=0; k < channels; k++){
				InputUnit *iu = new InputUnit();
				array[i][j][k]=iu;
				iu->layer = this;
				iu->lx=j;
				iu->ly=i;
				iu->lz=k;
			}
		}
	}

	return array;
}

Neuron* InputLayer::getInput(int, int, int, int, int, int){
	fprintf(stderr,"Trying to get input neuron from input neruon");
	assert(false);

	return NULL;
}

void InputLayer::setInputValue(int x, int y, int z, float value){
	InputUnit* n = dynamic_cast<InputUnit*>(neurons[y][x][z]);
	n->setInputValue(value);
}

float InputLayer::getInputValue(int x, int y, int z){
	InputUnit* n = dynamic_cast<InputUnit*>(neurons[y][x][z]);
	return n->getInputValue();
}

void InputLayer::backProp(int){}

void InputLayer::updateWeights(int){}

float InputLayer::getLamdba(){return epsilon*weightDecay;}

InputLayer::InputLayer(int width,int height, int channels){
	this->width = width;
	this->height = height;
	this->depth = channels;
	this->activationGradient = ActivationFunctions::noGradient;
	this->activationFunction = ActivationFunctions::noGradient;

	neurons = initNeuronArray(width,height, channels);

	printf("created input layer\n");
}
