#include "layer.h"

class OutputLayer {
	public:
		float (*regressionGradient) (float, float);
		float (*regressionError) (float, float);
		Layer* lastLayer;

		int currentPass,backPropPass;

		float*** errorGradients;

		bool useSoftmax;

		void backProp(float*** answers);

		void updateWeights();

		/**
		 * Forward propagates through the network.
		 * Assumes you have already set a new input example on the first layer
		 * @Returns the regression error sum from all inputs
		 */
		float forwardProp(float*** answers, bool printDiff);

		void initErrorGradientsArray(int width, int height, int depth);

		OutputLayer(Layer* previousLayer, float(*regError)(float,float), float (*regGradient) (float, float), bool useSoftmax);
};

