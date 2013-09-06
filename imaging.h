namespace imaging{
	float ** getImagePixels(const char* fileName, int desiredWidth, int desiredHeight);

	float ** readAnswers(const char * fileName, int width, int height);

	float *** generateAnswers(const char * fileName,bool printDebug, bool printOutput, int desiredWidth, int desiredHeight);
	
	void showKernals(float**** kernals, float* kernalBias, int kernalCount, int width, int height, int depth, int neuronNo, int setno, int epochno,bool printKernal);
	
	float *** generateAnswers(int x1, int y1, int x2, int y2, bool printDebug, bool printOutput, int colC, int rowC, int desiredWidth, int desiredHeight);
}
