#ifndef DNN_LAYER_H
#define DNN_LAYER_H 1

#define DEFAULT_EPSILON .0005
#define DEFAULT_WEIGHT_DECAY 0
#define DEFAULT_MOMENTUM .1

class Neuron;

class Layer{
	public:
		int width,height,depth;
		Neuron**** neurons;
		Layer * previousLayer;
		
		//lx,lz,lz (short for locationX) is the position in side the layer of the calling neuron
		//ix,iy,iz is where you are looking for
		virtual Neuron* getInput(int lx, int ly, int lz, int ix, int iy, int iz)=0;

		float epsilon;//learn rate
		float weightDecay;

		float momentum;//From 0 to 1, defines how much momentum will be kept;

		float (*activationFunction) (float);
		float (*activationGradient) (float);

		virtual void backProp(int) =0;
		virtual void updateWeights(int) = 0;

};

#endif
