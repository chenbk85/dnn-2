#include "ConvolutionalLayer.h"
#include "StandardUnit.h"
#include "activationFunctions.h"

#include <cstdio>
#include <algorithm>
#include <random>

using namespace std;

float ConvolutionalLayer::initKernal(){//choose a random kernal #
	return normal(generator);//FIX-------
}

float ConvolutionalLayer::initBias(){
	return 0;//Fix------------
}

void ConvolutionalLayer::updateWeights(int pass){
	for(int i=0; i <depth;i++){//which kernal
		float ***kernal = kernals[i];
		for(int y=0;y<kernalHeight;y++){
			for(int x=0;x<kernalWidth;x++){
				//if(y==0 && x==0)
				//	printf("Changing kernal %d %d %d from %f by %f to",i,y,x,*(kernal[y][x]),deltaKernal[i][y][x]);
				*(kernal[y][x]) = *(kernal[y][x]) + deltaKernal[i][y][x];
				deltaKernal[i][y][x]=0;
				//if(x==0 && y==0)
				//	printf(" %f\n",*(kernal[y][x]));
			}
		}

		//printf("chaning bias by %f from %f to ",deltaBias[i],biases[i]);
		biases[i]+=deltaBias[i];
		//printf("%f\n",biases[i]);
		deltaBias[i]*=momentum;
	}

	previousLayer->updateWeights(pass);
}

void ConvolutionalLayer::backProp(int pass){
	//printf("backprop conv %d %d %d\n",depth,height,width);
	
	for(int i=0; i <depth;i++){//which kernal
		float biasZError=0;
		for(int j=0; j <height;j++){//which row
			for(int k =0; k<width;k++){//which column
				//if(!(j==1 && k==0))continue;
				Neuron* n = neurons[j][k][i];//neuron row j, column k, kernal i
				float errorZGradient = n->errorZGradient;
				//if(i==0 && j==0 && k==0)
				//printf("Got conv neuron at (%d,%d,%d) with errorZGradient %f output %f %ld\n",j,k,i,errorZGradient, n->getOutput(pass), n);

				for(int y=0;y<n->iy;y++){
					for(int x=0;x<n->ix;x++){
						for(int z = 0; z < n->iz;z++){
							Neuron* input = n->getInput(y,x,z);
							//printf("got neuron %ld\n",input);
							float weight = n->weights[y][x][z];
							//printf("trying to add to error");
							input->addToYError(errorZGradient * weight,pass);
							//printf("added to erro\n");
							//
							float errorW = input->getOutput(pass)*errorZGradient;
							//printf("errorW %f\n",errorW);
							deltaKernal[i][y][x] += -epsilon*weightDecay*weight -epsilon*errorW;// = //INSERT//////////////////////////////////////////////
							//if(i==0 && j==0&&k==0)
							//printf("\nadded to delta %d %d %d %f out %f errorz %f errorW %f\n",i,y,x,deltaKernal[i][y][x], input->getOutput(pass), errorZGradient,errorW);
						}
					}
				}

				//printf("adding bias error: %f\n",errorZGradient);
				biasZError += errorZGradient;
			}
		}

		//printf("ready for bias\n");
		//printf("total bias error %f bias change %f\n",biasZError,-epsilon*biasZError);

		float biasErrorW = biasZError;//Because Y_bias = 1;
		float biasDeltaW = -epsilon*biasErrorW;
		deltaBias[i]+=biasDeltaW;
	}

	//printf("ging to previous layer\n");
	previousLayer->backProp(pass);
}

Neuron**** ConvolutionalLayer::initNeuronArray(int width, int height, int depth, int kernalWidth, int kernalHeight){
	Neuron**** array = new Neuron*** [height];
	for(int i =0; i<height;i++){
		array[i] = new Neuron** [width];
		for(int j = 0; j < width; j++){
			array[i][j] = new Neuron*[depth];
			for(int k = 0; k < depth;k++){
				StandardUnit *cn = new StandardUnit;
				array[i][j][k] = cn;

				cn->layer = this;
				cn->lx = j;
				cn->ly = i;
				cn->lz = k;

				cn->ix = kernalWidth;
				cn->iy = kernalHeight;
				cn->iz = previousLayer->depth;
				cn->weights = kernals[k];
				cn->bias = biases+k;
				//array[i][j][k].initInputArray(kernalWidth,kernalHeight,previousLayer->depth);
			}
		}
	}

	return array;
}

float **** ConvolutionalLayer::initKernalsArray(int width, int height, int depth, int kernalCount){
	float**** array = new float*** [kernalCount];
	printf("kernal: %d %d %d %d\n",kernalCount,height,width,depth);
	for(int i =0; i<kernalCount;i++){
		array[i] = new float** [height];
		for(int j = 0; j < height; j++){
			array[i][j] = new float* [width];
			for(int k = 0; k < width; k++){
				array[i][j][k] = new float[depth];
				for(int l = 0; l < depth; l++){
					array[i][j][k][l] = initKernal();
				}
			}
		}
	}

	return array;
}

Neuron* ConvolutionalLayer::getInput(int lx, int ly, int lz, int ix, int iy, int iz){
	//printf("getting %d %d %d %d %d %d\n",ly,lx,iz,iy,ix,ly+iy,lz+ix);
	int rowY = std::min(ly*kernalStride,maxY);
	int columnX = std::min(lx*kernalStride,maxX);
	//printf("row %d column %d arow %d acol %d %ld %ld\n",rowY,columnX, rowY+iy,columnX+ix,previousLayer, previousLayer->neurons[rowY+iy][columnX+ix][iz]);
	Neuron *n = previousLayer->neurons[rowY+iy][columnX+ix][iz];
	return n;
}

float * ConvolutionalLayer::initBiasArray(int kernalCount){
	float * array = new float[kernalCount];
	for(int i=0; i < kernalCount; i++)
		array[i]=initBias();
	return array;
}

float*** ConvolutionalLayer::initDeltaKernal(){
	float*** deltaKernal = new float **[depth];
	for(int i=0; i <depth; i++){
		deltaKernal[i] = new float*[kernalHeight];
		for (int y = 0; y < kernalHeight;y++){
			deltaKernal[i][y] = new float[kernalWidth];
			for(int x = 0; x<kernalWidth;x++){
				deltaKernal[i][y][x]=0;
			}
		}
	}

	return deltaKernal;
}

float* ConvolutionalLayer::initDeltaBias(){
	float* deltaBias = new float[depth];
	for(int i=0; i <depth;i++)
		deltaBias[i]=0;

	return deltaBias;
}

void ConvolutionalLayer::calculateDimentions(){
	printf("kernalStride: %d\n",kernalStride);
	maxX = previousLayer->width-kernalWidth;
	this->width = maxX/kernalStride+1;
	if(maxX%kernalStride!=0)
		width++;
	printf("width: %d\n",width);

	maxY = previousLayer->height-kernalHeight;
	this->height = maxY/kernalStride+1;
	if(height%kernalStride!=0)
		height++;
}

ConvolutionalLayer::ConvolutionalLayer(int kernalCount, int kernalWidth, int kernalHeight, int kernalStride, Layer* previousLayer, float (*activationFuncation) (float), float (*activationGradient) (float)){
	if(kernalWidth%2!=1 || kernalHeight%2!=1){
		fprintf(stderr,"Conv kernal size is not odd\n");
		return;
	}

	printf("create conv layer\n");

	int rand = (int)this;
	normal = std::normal_distribution<float>(0,(float).1);
	generator = std::default_random_engine (rand);

	epsilon = (float)DEFAULT_EPSILON;
	weightDecay = (float)DEFAULT_WEIGHT_DECAY;
	momentum = (float)DEFAULT_MOMENTUM;

	this->activationFunction = activationFuncation;
	this->activationGradient = activationGradient;

	this->kernalStride = kernalStride;
	this->depth = kernalCount;
	this->kernalWidth = kernalWidth;
	this->kernalHeight = kernalHeight;
	this->previousLayer = previousLayer;
	printf("calculating dimentions\n");
	calculateDimentions();

	printf("creating conv neurons\n");

	kernals = initKernalsArray(kernalWidth,kernalHeight,previousLayer->depth,depth);
	printf("creating bias array\n");
	biases = initBiasArray(kernalCount);
	printf("creating neuron array\n");
	neurons = initNeuronArray(width,height,kernalCount,kernalWidth, kernalHeight);
printf("creating delta kernal\n");
	deltaKernal = initDeltaKernal();
printf("creating delta bias\n");
	deltaBias = initDeltaBias();

	printf("done creating convlution layer\n");
}
