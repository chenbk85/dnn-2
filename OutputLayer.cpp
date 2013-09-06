#include "OutputLayer.h"
#include "neuron.h"

#include <cmath>
#include <cstdio>

using namespace std;

#define logit(a) (log(a)-log(1-a))

void OutputLayer::backProp(float*** answers){
	float sumofExp = 0;

	if(useSoftmax){
		for(int i=0; i<lastLayer->height; i++){
			for(int j=0; j<lastLayer->width; j++){
				for(int k = 0; k < lastLayer->depth; k++){
					Neuron* n = lastLayer->neurons[i][j][k];
					sumofExp += exp(logit(n->getOutput(currentPass)));
				}
			}
		}
	}

	//printf("sum of exp: %f\n",sumofExp);
	for(int i=0; i<lastLayer->height; i++){
		for(int j=0; j<lastLayer->width; j++){
			for(int k = 0; k < lastLayer->depth; k++){
				Neuron* n = lastLayer->neurons[i][j][k];
				float output = n->getOutput(currentPass);
				//printf("output before %f exp %f sum %f logit %f after",output,exp(logit(output)), sumofExp, logit(output));
				if(useSoftmax)
					output=exp(logit(output))/sumofExp;
				//printf("%f\n",output);

				float toAdd = -regressionGradient(output,answers[i][j][k]);
				n->addToYError(toAdd,currentPass);
			}
		}
	}

	lastLayer->backProp(currentPass);
}

void OutputLayer::updateWeights(){
	lastLayer->updateWeights(currentPass);
}

/**
 * Forward propagates through the network.
 * Assumes you have already set a new input example on the first layer
 * @Returns the regression error sum from all inputs
 */
float OutputLayer::forwardProp(float*** answers, bool printDiff){
	int h = lastLayer->height;
	int w = lastLayer->width;
	int d = lastLayer->depth;
	
	currentPass++;
	float error = 0;
	float diff =0;
	for(int i=0; i<h; i++){
		for(int j=0; j<w; j++){
			for(int k = 0; k < d; k++){
				//printf("trying to get neuron %d %d %d\n",i,j,k);
				Neuron* n = lastLayer->neurons[i][j][k];
				float output = n->getOutput(currentPass);//Forces it to do forward prop
				error+=regressionError(answers[i][j][k],output);
				if(printDiff){
					diff+=output-answers[i][j][k];
					printf("diff h: %d w: %d d: %d diff: %f output: %f answer: %f\n",i,j,k,diff, output,answers[i][j][k]);
				}
			}
		}
	}

	return error;
}

void OutputLayer::initErrorGradientsArray(int width, int height, int depth){
	errorGradients = new float**[height];
	for(int i=0; i<height; i++){
		errorGradients[i] = new float*[width];
		for(int j = 0; j < width;j++){
			errorGradients[i][j] = new float[depth];
			for(int k =0; k < depth;k++){
				errorGradients[i][j][k] = 0;
			}
		}
	}
}

OutputLayer::OutputLayer(Layer* previousLayer, float(*regError)(float,float), float (*regGradient) (float, float), bool useSoftmax){
	this->useSoftmax = useSoftmax;
	lastLayer = previousLayer;
	regressionGradient = regGradient;
	regressionError = regError;
	currentPass=0;
	backPropPass=0;

	initErrorGradientsArray(lastLayer->width,lastLayer->height,lastLayer->depth);
}
