#include "layer.h"

using namespace std;

class InputLayer : public Layer {
	public:
		Neuron**** initNeuronArray(int width, int height, int channels);
		
		Neuron* getInput(int, int, int, int, int, int);

		void setInputValue(int x, int y, int z, float value);

		float getInputValue(int x, int y, int z);
		
		void backProp(int);

		void updateWeights(int);

		float getLamdba();

		InputLayer(int width,int height, int channels);
};
