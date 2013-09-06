#include "InputUnit.h"

void InputUnit::setInputValue(float value){
	this->value = value;
}

float InputUnit::getInputValue(){
	return value;
}

float InputUnit::getOutput(int){
	//printf("here\n");
	return value;
}
