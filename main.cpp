#include "layer.h"
#include "InputLayer.h"
#include "OutputLayer.h"
#include "FullyConnectedLayer.h"
#include "ConvolutionalLayer.h"
#include "regressionFunctions.h"
#include "activationFunctions.h"
#include "imaging.h"

#include <algorithm>
#include <vector>
#include <string>
#include <cstdio>
#include <iostream>
#include <opencv/cxcore.hpp>
#include "opencv/highgui.h"
//#include "c:\Users\Eric\opencv\include\opencv\highgui.h">
using namespace cv;

class DNN{/*
	     static const int numConvLayers = 5;
	     static const int numFCLayers = 2;
	     static const int inputWidth = 250;
	     static const int inputHeight = 250;
	     static const int outputWidth = 1000;
	     *//*
		  InputLayer iL;
		  ConvolutionLayer* convolutionLayers;
		  FullyConnectedLayer* fullyConnectedLayers;
		  OutputLayer oL;

		  DNN(){

		  }

		  ConvolutionLayer* buildConvLayers(int numLayers, int* kernalCounts, int* width, int* height){

		  }*/
};

struct InputSet{
	float *** input;
	int inputCount, inputHeight, inputWidth;
};

struct AnswerSet{
	float **** answers;
	int inputCount, answerHeight, answerWidth;
};

/*void test(float*** testSet, float ** answers){

}*/

void trainMiniBatch(InputSet* input, AnswerSet* output, InputLayer* inputLayer, OutputLayer* outputLayer, int loopCount, Layer** layers, int numLayers,ConvolutionalLayer *a,ConvolutionalLayer *b){
	float decay = .9;
	for(int i =0; i < loopCount; i++){
		if(i%50==0){
			//waitKey(0);
			if(a!=NULL)
				imaging::showKernals(a->kernals,a->biases,a->depth,a->kernalWidth,a->kernalHeight,a->previousLayer->depth,0,-1,i,true);
			if(b!=NULL)
				imaging::showKernals(b->kernals,b->biases,b->depth,b->kernalWidth,b->kernalHeight,b->previousLayer->depth,1,-1,i,true);
			//waitKey(0);
		}
		if(i%200==0){
			waitKey(0);
		}
		printf("\nError: ");
		for(int ind = 0; ind < input->inputCount; ind++){
			for(int j =0; j <input->inputHeight;j++){
				for(int k =0; k<input->inputWidth;k++)
				{
					inputLayer->setInputValue(k,j,0,input->input[ind][j][k]);
					//printf("setting input %f\n",input->input[ind][j][k]);
				}
			}

			float error = outputLayer->forwardProp(output->answers[ind],false);

			printf("%.2f ",error);
			outputLayer->backProp((output->answers)[ind]);
			//if(ind%5==0)
			outputLayer->updateWeights();
			
			/*
			   printf("BEFORE BACK PROP\n");

			   for(int num =0; num < numLayers-1; num++){
			   ConvolutionalLayer* cn = dynamic_cast<ConvolutionalLayer*>(layers[num]);
			   for(int num2 =0; num2 <cn->depth;num2++){

			   if(num==0 && num2==0){
			   printf("layer %d bias: %f one weight %f\n",num,cn->biases[num2],cn->kernals[num2][0][0][0]);
			   for(int num3 =0; num3 <10;num3++){
			   for(int num4 =0; num4 <10;num4++){
			   for(int num5=0; num5 < cn->previousLayer->depth;num5++)
			   printf("%f ",cn->kernals[num2][num3][num4][num5]);
			   }
			   printf("\n");
			   }
			   printf("\n");
			   }
			   }
			   }*/
			/*printf("AFTER BACKPROP\n");
			  for(int num =0; num < numLayers-1; num++){
			  ConvolutionalLayer* cn = dynamic_cast<ConvolutionalLayer*>(layers[num]);
			  for(int num2 =0; num2 <cn->depth;num2++)
			  printf("layer %d bias: %f one weight %f\n",num,cn->biases[num2],cn->kernals[num2][0][0][0]);
			  }*/
			if(ind%40==39)printf("\n");
		}
		printf("\nDone with EPOCH %d!\n",i);

		for(int num = 0; num < numLayers;num++){
			//layers[num]->epsilon*=decay;
		}

		//printf("New epsilon: %f\n",layers[0]->epsilon);
	}
}

int main(){
	
	const char* imagesListFileName = "C:\\Users\\Eric\\Downloads\\images\\a\\jpgimgs";
	const char* answersListFileName = "C:\\Users\\Eric\\Downloads\\images\\a\\answers";

	FILE *imagesListFile =fopen(imagesListFileName, "r");
	FILE *answersListFile = fopen(answersListFileName, "r");

	char fileName[100];
	vector<string> allFileNames;
	vector<string> allAnswersFileNames;
	while(fscanf(imagesListFile,"%s",fileName)!=EOF){
		allFileNames.push_back(string(fileName));
	}
	fclose(imagesListFile);

	while(fscanf(answersListFile,"%s",fileName)!=EOF){
		allAnswersFileNames.push_back(string(fileName));
	}
	fclose(answersListFile);

	bool autoGen = true;
	string fileNameBase = "C:\\Users\\Eric\\Downloads\\images\\a\\";

	int sizein = 30;
	int maxFiles = 2000;
	int numEx=allFileNames.size()>maxFiles?maxFiles:allFileNames.size();
	int numEpochs=1000;
	InputLayer iL(sizein,sizein,1);

	std::normal_distribution<float> normal(sizein/4,sizein/8);
	std::normal_distribution<float> normalLines(sizein/2,sizein/3);

	std::default_random_engine generator((int)(&iL));

	InputSet input;
	input.inputHeight = sizein;
	input.inputWidth = sizein;
	input.inputCount = numEx;
	input.input = new float**[numEx];
	vector<int> x1s,x2s,y1s,y2s;
	vector<bool> hasLines;
	
	if(autoGen){
		for(int ind = 0; ind < numEx; ind++){
			input.input[ind] = new float*[sizein];
			
			for(int i =0; i <sizein;i++){
				input.input[ind][i] = new float[sizein];
				for(int j =0; j<sizein;j++)
				{
					//input.input[ind][i][j] = .1;
					
					
					float val = normal(generator)/sizein;
					val = val<0?0.0f:(val>.4?.4f:val);
					//printf("%f ",val);
					input.input[ind][i][j] = val;
				}
				//printf("\n");
			}
			//printf("\n");

			float random = normal(generator);
			if(random>sizein/8)
				hasLines.push_back(true);
			else
				hasLines.push_back(false);

			if(hasLines[ind]){

				int x1 = (int)normalLines(generator);
				x1 = x1<0?0:(x1>(sizein-1)?(sizein-1):x1);
				int x2 = (int)normalLines(generator);
				x2 = x2<0?0:(x2>(sizein-1)?(sizein-1):x2);
				int y1 = (int)normalLines(generator);
				y1 = y1<0?0:(y1>(sizein-1)?(sizein-1):y1);
				int y2 = (int)normalLines(generator);
				y2 = y2<0?0:(y2>(sizein-1)?(sizein-1):y2);

				if(x2<x1){
					int temp = x2;
					x2 = x1;
					x1 = temp;
					temp = y2;
					y2=y1;
					y1 = temp;
				}

				printf("%d %d %d %d\n",x1,x2,y1,y2);

				x1s.push_back(x1);
				x2s.push_back(x2);
				y1s.push_back(y1);
				y2s.push_back(y2);


				for(int i=x1; i <= x2; i++){
					if(x1!=x2){
						int y = y1+(y2-y1)*(i-x1)/(x2-x1);
						y=y<0?0:(y>(sizein-1)?(sizein-1):y);
						int yn = y1+(y2-y1)*(i+1-x1)/(x2-x1);
						yn=yn<0?0:(yn>(sizein-1)?(sizein-1):yn);
						printf("i: %d y: %d yn: %d\n",i,y,yn);
						for(int j = y;j<=yn;j++)
							input.input[ind][j][i] = .8f;
						for(int j = y;j>=yn;j--)
							input.input[ind][j][i] = .8f;
					}
					else{
						for(int j = y1; j<=y2;j++){
							input.input[ind][j][i]=.8f;
						}
						for(int j = y1; j>=y2;j--){
							input.input[ind][j][i]=.8f;
						}
					}
				}
			}

			printf("input: %d hasLine: %d\n",ind,hasLines[ind]?1:0);
			/*
			for(int i =0;i < sizein;i++){
				for(int j=0; j<sizein;j++){
					if(input.input[ind][i][j]==.8f)
						printf("XXX ");
					else
						printf("%.1f ", input.input[ind][i][j]);
				}
				printf("\n");
			}
			printf("\n");*/
		}
		printf("input done\n");
	}
	else{
		printf("Num: %d\n",numEx);
		for(int ind = 0; ind<numEx; ind++){
			string completeFile = fileNameBase + allFileNames[ind];
			printf("Looking in file: %s\n",completeFile.c_str());
			
			input.input[ind] = imaging::getImagePixels(completeFile.c_str(), sizein,sizein);
			/*for(int i=0; i < sizein; i++){
				for(int j =0; j < sizein;j++){
					printf("%.1f ",input.input[ind][i][j]);
				}
				printf("\n");
			}
			printf("\n");*/
		}
	}

	int outHeight = 10, outWidth = 10, outDepth=1;//,sizeH = 200;
	Layer ** layers = new Layer*[4];
	ConvolutionalLayer  a(1,11,11,1,&iL,ActivationFunctions::logistic, ActivationFunctions::logisticGradient);
	//ConvolutionalLayer  b(1,5,5,1,&a,ActivationFunctions::logistic, ActivationFunctions::logisticGradient);
	//ConvolutionalLayer  c(20,3,3,1,&b,ActivationFunctions::logistic, ActivationFunctions::logisticGradient);
	//printf("h %d w %d depth: %d\n",c.height,c.width,c.depth);
	//FullyConnectedLayer b(outWidth*3,outHeight*3,&iL,ActivationFunctions::logistic, ActivationFunctions::logisticGradient);
	//FullyConnectedLayer c(outWidth*2,outHeight*2,&a,ActivationFunctions::logistic, ActivationFunctions::logisticGradient);
	//FullyConnectedLayer d(1,1,&c,ActivationFunctions::logistic, ActivationFunctions::logisticGradient);
	OutputLayer oL(&a,RegressionFunctions::squaredError, RegressionFunctions::squaredErrorGradient,false);
	//OutputLayer oL(&a,RegressionFunctions::softmaxError, RegressionFunctions::softmaxErrorGradient,true);

	Layer* lastLayer = &a;
	int oh = lastLayer->height;
	int ow = lastLayer->width;
	int od = lastLayer->depth;
	printf("output: %d %d %d\n",oh,ow,od);

	//layers[0] = &a;
	//layers[1] = &b;
	//layers[2] = &b;
	//layers[3] = &d;

	printf("Starting\n");

	AnswerSet as;
	float**** answers;
	answers = new float***[numEx];

	//normal(50,100);
	std::normal_distribution<float> normalAnswer(.5,.3);

	std::default_random_engine generatorAnswer((int)(&oL));

	if(autoGen){
		printf("answer\n");
		for(int i =0; i<numEx; i++){
			//answers[i] = imaging::generateAnswers(x1s[i],y1s[i],x2s[i],y2s[i],false,false,sizein,sizein,ow,oh);
			
			//printf("%d %d %d\n",oh,ow,od);
			answers[i] = new float**[oh];
			for(int j = 0; j <oh; j++){
				answers[i][j] = new float*[ow];
				for(int k = 0; k < ow; k++){
					answers[i][j][k]=new float[od];
					for(int m=0; m <od; m++){
						//answers[i][j][k][m] = hasLines[i]?1.0:0.0;

						int jmin = j<2?0:j-2;
						int kmin = k<2?0:k-2;
						//float a = input.input[i][j+5][k+5]/3.0f+input.input[i][j+3][k+3]/3.0f+input.input[i][j+3][k+7]/3.0f;//jmin][kmin];//normalAnswer(generatorAnswer);
						float a = input.input[i][j+5][k+5];
						if(a<.5)a=0;
						//printf("%f ",a);
						a = a>1.0?1.0:(a<0.0?0.0:a);
						answers[i][j][k][m] = a;
						//printf("got %f\n",answers[i][j][k][0]);
					}
					
				}
				//printf("\n");
				//printf("set %f\n",answers[i][j][0]);
			}
			//printf("\n");
		}
	}
	else{
		for(int ind = 0;ind < numEx;ind++){
			string completeFile = fileNameBase + allAnswersFileNames[ind];
			printf("Generating answers for: %s\n",completeFile.c_str());
			answers[ind] = imaging::generateAnswers(completeFile.c_str(),false,true,ow,oh);
		}
	}

	as.answers = answers;
	as.inputCount = numEx;
	as.answerHeight = outHeight;
	as.answerWidth = outWidth;

	trainMiniBatch(&input, &as, &iL, &oL, numEpochs, layers, 4, &a, NULL);

	return 0;
}
