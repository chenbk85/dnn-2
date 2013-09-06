#include "FullyConnectedLayer.h"
#include "activationFunctions.h"
#include "StandardUnit.h"

#include <algorithm>
#include <cstdio>
#include <random>

using namespace std;

void FullyConnectedLayer::backProp(int pass){
	for(int i=0; i <height;i++){
		for(int j=0; j <width;j++){
			Neuron* n = neurons[i][j][0];

			int ph = n->iy;
			int pw = n->ix;
			int pd = n->iz;

			//printf("got neuron at %d %d looking through: %d %d %d\n",i,j,ph,pw,pd);

			float errorZGradient = n->errorZGradient;
			//printf("errorZ %d %d %f\n",i,j,errorZGradient);
			for(int y=0;y<ph;y++){
				for(int x=0;x<pw;x++){
					for(int z = 0; z < pd;z++){
						Neuron* input = n->getInput(y,x,z);
						float output = input->getOutput(pass);
						float weight = n->weights[y][x][z];
						input->addToYError(errorZGradient * weight,pass);
						float errorW = weightDecay*weight+output*errorZGradient;
						float deltaW = -epsilon*errorW;// = //INSERT//////////////////////////////////////////////
						n->deltaWeights[y][x][z]+=deltaW;
						//if(y==0 && x==0 && z==0)printf("%d %d %d changing the weight by: %f to %f output %f epsilon %f errorz %f\n",y,x,z,deltaW,n->weights[y][x][z], output, epsilon, errorZGradient);
					}
				}
			}

			float deltaBias = -epsilon*errorZGradient;
			//printf("changing bias by %f \n",deltaBias);
			//printf("deltaBias %d %d %f z %f\n",i,j,deltaBias,errorZGradient);
			*(n->deltaBias)+=deltaBias;
		}
	}

	//printf("to next layer\n");

	previousLayer->backProp(pass);
}

void FullyConnectedLayer::updateWeights(int pass){
	for(int i=0; i <height;i++){
		for(int j=0; j <width;j++){
			Neuron* n = neurons[i][j][0];
			
			int ph = n->iy;
			int pw = n->ix;
			int pd = n->iz;

			for(int y=0;y<ph;y++){
				for(int x=0;x<pw;x++){
					for(int z = 0; z < pd;z++){
						n->weights[y][x][z]+=n->deltaWeights[y][x][z];
						n->deltaWeights[y][x][z]*=momentum;
					}
				}
			}

			*(n->bias) += *(n->deltaBias);
			*n->deltaBias = 0;
		}
	}

	previousLayer->updateWeights(pass);
}

Neuron**** FullyConnectedLayer::initNeuronArray(int width, int height){
	Neuron**** array = new Neuron*** [height];
	for(int i =0; i<height;i++){
		array[i] = new Neuron** [width];
		for(int j = 0; j < width; j++){
			array[i][j] = new Neuron*[1];
			StandardUnit* fcu = new StandardUnit();
			array[i][j][0] = fcu;
			
			fcu->bias = new float;
			*(fcu->bias) = initBias();
			fcu->deltaBias = new float;
			*(fcu->deltaBias) = 0;

			fcu->weights = initWeightArray(previousLayer->width,previousLayer->height, previousLayer->depth);
			fcu->deltaWeights = initDeltaWeightArray(previousLayer->width,previousLayer->height, previousLayer->depth);
			fcu->layer = this;

			fcu->lx = j;
			fcu->ly = i;
			fcu->lz = 0;

			fcu->ix = previousLayer->width;
			fcu->iy = previousLayer->height;
			fcu->iz = previousLayer->depth;
		}
	}

	printf("created fully connected array\n");

	return array;
}

float*** FullyConnectedLayer::initDeltaWeightArray(int width, int height,int depth){
	float*** array = new float** [height];
	for(int i =0; i<height;i++){
		array[i] = new float* [width];
		for(int j = 0; j < width; j++){
			array[i][j] = new float[depth];
			for(int k = 0; k < depth; k++){
				array[i][j][k] = 0;
			}
		}
	}

	return array;
}

float*** FullyConnectedLayer::initWeightArray(int width, int height,int depth){
	float*** array = new float** [height];
	for(int i =0; i<height;i++){
		array[i] = new float* [width];
		for(int j = 0; j < width; j++){
			array[i][j] = new float[depth];
			for(int k = 0; k < depth; k++){
				array[i][j][k] = initWeight();
			}
		}
	}

	return array;
}

float FullyConnectedLayer::initBias(){
	return 0;//FIX------------
}

float FullyConnectedLayer::initWeight(){
	//unsigned int b = eng();
	float a = normal(generator);
	//printf("got float: %f\n",a);
	return a;
}

Neuron* FullyConnectedLayer::getInput(int, int, int, int ix, int iy, int iz){
	return previousLayer->neurons[iy][ix][iz];
}

//depth assumed to be one
FullyConnectedLayer::FullyConnectedLayer(int width,int height, Layer* previousLayer, float (*activationFunction) (float),float (*actGrad)(float)){
	int rand = (int)previousLayer;
	normal = std::normal_distribution<float>(0,(float).1);
	generator = std::default_random_engine (rand);

	printf("creating fully connected layer\n");
	epsilon = (float)DEFAULT_EPSILON;
	momentum = (float)DEFAULT_MOMENTUM;

	this->activationFunction = activationFunction;
	this->activationGradient = actGrad;
	this->activationGradient(0);
	this->width = width;
	this->height = height;
	this->depth = 1;
	this->previousLayer = previousLayer;

	neurons = initNeuronArray(width,height);

	//connectLayer();
	printf("done connecting fully connected layer\n");
}
