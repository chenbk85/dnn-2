#include "neuron.h"
#include "layer.h"

Neuron::Neuron(){
	lastBackPropPass = -1;
}
Neuron* Neuron::getInput(int y, int x, int z){
	Neuron *n = layer->getInput(lx,ly,lz,x,y,z);
	return n;
}

void Neuron::addToYError(float toAdd, int pass){
	if(pass!=lastBackPropPass){
		lastBackPropPass = pass;
		errorYGradient = 0;
		errorZGradient = 0;
	}

	errorYGradient+= toAdd;
	float output = getOutput(pass);
	float grad = layer->activationGradient(output);
	//printf("%d %d %d adding %f to y grad. %f to z grad\n",lx,ly,lz,toAdd,toAdd*grad);
	errorZGradient+= toAdd * grad;
}

/*
   class MaxPooling : public Neuron{
   public:
   float getOutput(int pass){
   float output = 0;
   for(int i =0; i < ix; i++){
   for(int j =0; j < iy; j++){
   for(int k =0; k < iz; k++){
   output = inputs[i][j][k]->getOutput()>output?inputs[i][j][k]->getOutput():output;
   }
   }
   }

   return output;
   }
   };*/
/*
   class Unit : public Neuron{
   private:
   int passCount;
   float outputComputed;
   public:
   float *kernalBias;

   ConvUnit(){
   passCount = -1;
   }

   float getOutput(int pass){
//printf("begin\n");

if(pass==passCount)
return outputComputed;//don't compute twice
passCount =pass;

float output = 0;
for(int i=0; i <iy;i++){
for(int j=0; j<ix;j++){
for(int k=0;k <iz;k++){
//printf("trying to get: %d %d %d\n",i,j,k);
Neuron* n = getInput(i,j,k);
//printf("%d %d %d kernal %f\n",i,j,k,weights[i][j][k]);
float output = n->getOutput(pass);
//printf("output %f\n",output);
output+=output*weights[i][j][k];
}
}
}
//printf("run activationFunction\n");

outputComputed = layer->activationFunction(output+*kernalBias);
//printf("computed: %lf\n",outputComputed);

return outputComputed;
}
};*/


/*
   class LocalResponseNormalization : public Neuron{
   public:
   float alpha, k, beta;

   float getOutput(){
   float output=inputs[0], sum=0;
   for(int i=0; i<inputCount; i++){
   sum+=inputs[i].getOutput()*inputs[i].getOutput();
   }

   sum= pow(sum*alpha+k,beta);
   output/=sum;
   return output;
   }
   };*/
