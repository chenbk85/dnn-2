#include "StandardUnit.h"
#include "Layer.h"

#include <cstdio>
using namespace std;

StandardUnit::StandardUnit(){
	passCount = -1;
}

float StandardUnit::getOutput(int pass){
	if(pass==passCount)
		return outputComputed;

	passCount = pass;
	float output = 0;
	float weights2 = 0;
	//printf("start %d %d %d\n",ly,lx,lz);
	for(int i=0; i <iy;i++){
		for(int j=0; j<ix;j++){
			for(int k=0;k <iz;k++){
				//printf("trying to get input\n");
				Neuron* n = getInput(i,j,k);
				//printf("got input %ld\n",n);
				output+=n->getOutput(pass)*weights[i][j][k];
				weights2+=weights[i][j][k]*weights[i][j][k];
				//printf("%f ",n->getOutput(pass));
			}
		}
		//printf("\n");
	}
	//printf("\n");
	outputComputed = layer->activationFunction(output+*bias)+weights2*(layer->weightDecay)/2.0;
	return outputComputed;
}
