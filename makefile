OPENCV_DIR=%OPENCV_x64_DIR%
OUTPUT_FILE=dnn.exe

PARAMS=/W4 /EHsc /O2 /c
PARAMS_OPENCV=/I $(OPENCV_INCLUDE) $(PARAMS)
PARAMS_LINK=/LIBPATH:$(OPENCV_DIR)\lib

OPENCV_LIBS=opencv_core243.lib opencv_imgproc243.lib opencv_highgui243.lib opencv_ml243.lib opencv_video243.lib opencv_features2d243.lib opencv_calib3d243.lib opencv_objdetect243.lib opencv_contrib243.lib opencv_legacy243.lib opencv_flann243.lib opencv_nonfree243.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib

all: neuron.obj layer.obj activationFunctions.obj regressionFunctions.obj InputUnit.obj StandardUnit.obj InputLayer.obj OutputLayer.obj FullyConnectedLayer.obj ConvolutionalLayer.obj main.obj imaging.obj
	link /OUT:$(OUTPUT_FILE)  $(PARAMS_LINK) $(OPENCV_LIBS) neuron.obj layer.obj activationFunctions.obj InputUnit.obj StandardUnit.obj InputLayer.obj OutputLayer.obj FullyConnectedLayer.obj ConvolutionalLayer.obj regressionFunctions.obj main.obj imaging.obj

main.obj: main.cpp
	cl $(PARAMS_OPENCV) main.cpp
neuron.obj: neuron.cpp
	cl  $(PARAMS) neuron.cpp
InputUnit.obj: InputUnit.cpp
	cl  $(PARAMS) InputUnit.cpp
StandardUnit.obj: StandardUnit.cpp
	cl  $(PARAMS) StandardUnit.cpp
InputLayer.obj: InputLayer.cpp
	cl  $(PARAMS) InputLayer.cpp
OutputLayer.obj: OutputLayer.cpp
	cl  $(PARAMS) OutputLayer.cpp
FullyConnectedLayer.obj: FullyConnectedLayer.cpp
	cl  $(PARAMS) FullyConnectedLayer.cpp
ConvolutionalLayer.obj: ConvolutionalLayer.cpp
	cl  $(PARAMS) ConvolutionalLayer.cpp
layer.obj: layer.cpp
	cl $(PARAMS) layer.cpp
activationFunctions.obj: activationFunctions.cpp
	cl $(PARAMS) activationFunctions.cpp
regressionFunctions.obj: regressionFunctions.cpp
	cl $(PARAMS) regressionFunctions.cpp
imaging.obj: imaging.cpp
	cl $(PARAMS_OPENCV) imaging.cpp
clean:
	del *.obj
