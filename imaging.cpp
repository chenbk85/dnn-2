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

	void showKernals(float**** kernals, float* kernalBias, int kernalCount, int width, int height, int depth, int neuronNo,int setno, int epochno,bool showKernals){
		for(int ind = 0; ind <kernalCount;ind++){
			for(int d = 0; d <depth; d++){
				Mat image;
				image.create(500,500,CV_32FC1);
				int spanx=500/width,spany=500/height;
				for(int i=0; i <height; i++){
					
					for(int j=0; j <width; j++){
						if(showKernals)printf("%f ",kernals[ind][i][j][d]);
						for(int x =0; x < spanx; x++)
							for(int y=0;y<spany;y++){	
								image.at<float>(i*spany+y,j*spanx+x) = abs(kernals[ind][i][j][d]);
							}
						
					}
					if(showKernals)printf("\n");
				}
				//if(showKernals)printf("Bias: %f\n\n",kernalBias[ind]);
				printf("\n");

				stringstream ss;
				ss<<"Image Neuron No: "<<neuronNo<<" Kernal No: "<<ind<<" depth: "<<d<<" set: "<<setno<<" epoch: "<<epochno;
				imshow(ss.str(),image);
			}
		}
		
	}

	float ** getImagePixels(const char* fileName, int desiredWidth, int desiredHeight){
		Mat src = imread(fileName, CV_LOAD_IMAGE_GRAYSCALE);
		printf("rows: %d cols:%d\n",src.rows,src.cols);
		float ** vals = new float*[desiredHeight];
		for(int i =0;i < desiredHeight; i++){
			vals[i] = new float[desiredWidth];
			for(int j =0; j<desiredWidth;j++){
				vals[i][j] = 0;
			}
		}

		int minR = 60, maxR = src.rows-minR;
		int minC = 40, maxC = src.cols-minC;
		float area = (float)(maxR-minR)*(maxC-minC);

		for(int i =minR;i < maxR; i++){
			for(int j =minC;j < maxC; j++){
				int row = i*desiredHeight/src.rows;
				int column = j * desiredWidth/src.cols;

				float val = (float)src.at<char>(i,j);
				//printf("got val: %f\n",val);
				float outVal = val/255.0f*desiredHeight*desiredWidth/area;
				outVal = min(outVal,1.0f);
				outVal = max(outVal,0.0f);
				//printf("outval: %f\n",outVal);
				/*
				if(outVal<-.35)printf("X");
				else printf("O");*/
				vals[row][column] += outVal;
			}
			//printf("\n");
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

	float *** generateAnswers(int x1, int y1, int x2, int y2, bool printDebug, bool printOutput, int colC, int rowC, int desiredWidth, int desiredHeight){
		int numRadiusBins = desiredWidth;
		int numAngleBins = desiredHeight;
		int maxRadius = rowC/2+colC/2;

		float *** vals = new float**[numAngleBins];
		for(int i=0; i < numAngleBins; i++){
			vals[i] = new float*[numRadiusBins];
			for(int j=0; j<numRadiusBins;j++){
				vals[i][j] = new float[1];
				vals[i][j][0] = 0;
			}
		}

		int a = x1,b=y1,c=x2,d=y2;

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

		float dist = sqrt((c-a)*(c-a)+(d-b)*(d-b));
		if(dist<1){
			dist=1;
		}

		float r = ((c-a)*b-a*(d-b))/dist;
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

		
		//printf("des %d %d\n",desiredHeight, desiredWidth);
		if(printDebug)
			printf("bin r %d t %d\n",radiusBin,angleBin);

		vals[angleBin][radiusBin][0]=.5;

		if(printOutput){
			printf("printing answer\n");
		for(int i =0; i<numAngleBins;i++){
			for(int j=0; j<numRadiusBins;j++){
				printf("%.2f ",vals[i][j][0]);
			}
			printf("\n");
		}
		printf("\n");
		}

		return vals;
	}

	float *** generateAnswers(const char * fileName,bool printDebug, bool printOutput, int desiredWidth, int desiredHeight){
		FILE * file = fopen(fileName,"r");
		int a,b,c,d;
		int rowC = 240;
		int colC = 320;

		int numRadiusBins = desiredWidth;
		int numAngleBins = desiredHeight;
		int maxRadius = 320;

		float *** vals = new float**[numAngleBins];
		for(int i=0; i < numAngleBins; i++){
			vals[i] = new float*[numRadiusBins];
			for(int j=0; j<numRadiusBins;j++){
				vals[i][j] = new float[1];
				vals[i][j][0] = 0;
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

			float dist = sqrt((c-a)*(c-a)+(d-b)*(d-b));
			if(dist<1){
				num--;
				continue;
			}
			
			float r = ((c-a)*b-a*(d-b))/dist;
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

			vals[angleBin][radiusBin][0] ++;
		}

		for(int i=0; i < numAngleBins; i++){
			for(int j=0; j<numRadiusBins;j++){
				if(num!=0)
					vals[i][j][0]/=num;
				if(printOutput)
					printf("%f ",vals[i][j][0]);
			}
			if(printOutput)
				printf("\n");
		}
		if(printOutput)
			printf("\n");

		fclose(file);
		return vals;
	}

}
