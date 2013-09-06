#include "imaging.h"

#include <cstdio>
using namespace std;

#include "opencv2\calib3d\calib3d.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"

using namespace cv;

namespace imaging{

	float ** getImagePixels(const char* fileName, int desiredWidth, int desiredHeight){
		Mat src = imread(fileName, 0);
		float ** vals = new float*[desiredHeight];
		for(int i =0;i < desiredHeight; i++){
			vals[i] = new float[desiredWidth];
			for(int j =0; j<desiredWidth;j++){
				vals[i][j] = 0;
			}
		}

		float areaRatio = src.rows*src.cols/((float)desiredWidth*desiredHeight);
		float divisor = 255.0f * areaRatio;
		for(int i =0;i < src.rows; i++){
			for(int j =0;j < src.cols; j++){
				int row = i*desiredHeight/src.rows;
				int column = j * desiredWidth/src.cols;

				vals[row][column] +=(src.at<float>(i,j)-128.0f)/divisor;
			}
		}

		return vals;
	}

	float ** readAnswers(const char * fileName, int width, int height){
		float ** vals = new float*[height];
		FILE * file = fopen(fileName,"r");
		for(int i =0;i < height; i++){
			vals[i] = new float[width];
			for(int j = 0; j <width; j++){
				fscanf(file,"%f\n",vals[i]+j);
			}
		}

		fclose(file);
		return vals;
	}

	float ** generateAnswers(const char * fileName,bool printDebug, bool printfOutput){
		FILE * file = fopen(fileName,"r");
		int a,b,c,d;
		int rowC = 240;
		int colC = 320;

		int numRadiusBins = 18;
		int numAngleBins = 36;
		int maxRadius = 320;

		float ** vals = new float*[numAngleBins];
		for(int i=0; i < numAngleBins; i++){
			vals[i] = new float[numRadiusBins];
			for(int j=0; j<numRadiusBins;j++){
				vals[i][j] = 0;
			}
		}

		float num=0;
		while(fscanf(file,"%d %d %d %d",&a,&b,&c,&d)!=EOF){
			num++;

			if(c<a || (c==a && d <b)){
				int temp =c;
				c = a;
				a = temp;
				temp = b;
				b = d;
				d = temp;
			}

			if(printDebug)
				printf("orig %d %d %d %d\n",a,b,c,d);

			a-=colC;
			c-=colC;

			b=rowC-b;
			d=rowC-d;

			if(printDebug)
				printf("got %d %d %d %d\n",a,b,c,d);

			float r = ((c-a)*b-a*(d-b))/sqrt((c-a)*(c-a)+(d-b)*(d-b));
			float theta = 3.14159f/2.0f+atan2(d-b,c-a);

			if(printDebug)
				printf("got angle: %f %f %d %d\n",theta,atan2(d-b,c-a),d-b,c-a);


			if(theta<0)
				theta+=3.14159f*2.0f;
			if(theta>3.14159f){
				theta-=3.14159f;
				r*= -1;
			}

			if(printDebug)
				printf("r %f t %f\n",r,theta);

			theta/=3.14159f;
			theta = min(theta,1.0f);
			theta = max(theta,0.0f);

			r/=maxRadius*2.0f;
			r = min(r,.5f);
			r = max(r,-.5f);
			r+=.5f;

			int angleBin = (int)(theta*(numAngleBins-1));
			int radiusBin = (int)(r*(numRadiusBins-1));

			if(printDebug)
				printf("bin r %d t %d\n",radiusBin,angleBin);

			vals[angleBin][radiusBin] ++;
		}

		for(int i=0; i < numAngleBins; i++){
			for(int j=0; j<numRadiusBins;j++){
				vals[i][j]/=num;
				if(printOutput)
					printf("%f ",vals[i][j]);
			}
			if(printOutput)
				printf("\n");
		}

		fclose(file);
		return vals;
	}

}
