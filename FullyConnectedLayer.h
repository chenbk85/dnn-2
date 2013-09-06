#include "layer.h"
#include <random>

using namespace std;

class FullyConnectedLayer : public Layer {
	public:
		Neuron* getInput(int, int, int, int ix, int iy, int iz);

		//depth assumed to be one
		FullyConnectedLayer(int width,int height, Layer* previousLayer, float (*activationFunction) (float),float (*actGrad)(float));
		
		void backProp(int pass);

		void updateWeights(int pass);
		
		float getLamdba();
	
	private:
		std::normal_distribution<float> normal;//(0,.1);//(3.0, 4.0);
		std::default_random_engine generator;

		Neuron**** initNeuronArray(int width, int height);
		
		float*** initWeightArray(int width, int height,int depth);
		float*** initDeltaWeightArray(int width, int height,int depth);

		float initBias();

		float initWeight();
};

