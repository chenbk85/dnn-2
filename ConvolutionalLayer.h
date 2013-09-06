#include "layer.h"

#include <random>

using namespace std;

class ConvolutionalLayer : public Layer {
	public:
		Neuron* getInput(int lx, int ly, int lz, int ix, int iy, int iz);
		
		void backProp(int pass);
		void updateWeights(int pass);
		
		ConvolutionalLayer(int kernalCount, int kernalWidth, int kernalHeight, int kernalStride, Layer* previousLayer, float (*activationFuncation) (float), float (*activationGradient) (float));
		float **** kernals, *** deltaKernal;
		int kernalWidth, kernalHeight, kernalStride;
		float * biases, * deltaBias;
		
	private:
		float **** kernalMomentum;

		int maxX, maxY;
		
		std::normal_distribution<float> normal;
		std::default_random_engine generator;
		
		float initKernal();
		
		float initBias();
		
		Neuron**** initNeuronArray(int width, int height, int depth, int kernalWidth, int kernalHeight);

		float **** initKernalsArray(int width, int height, int depth, int kernalCount);

		float *** initDeltaKernal();

		float * initBiasArray(int kernalCount);

		float * initDeltaBias();

		void calculateDimentions();
};
