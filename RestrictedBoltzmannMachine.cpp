#include <sstream> //stringstream
#include <string> //string
#include <iomanip> //setprecision
#include <fstream> //ofstream
#include <cmath> //ceil
#include <algorithm> //transform
#include <cblas.h>
#include "Utility.h"
#include "RestrictedBoltzmannMachine.h"

#include <iostream>

void RestrictedBoltzmannMachine::SampleFromHiddenGivenVisible(double* hiddenMatrix, const double* visibleMatrix, const double* resizedHiddenBiases, const int samples) const {
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, samples, numHidden, numVisible, 1, visibleMatrix, samples, weightsMatrix, numVisible, 0, hiddenMatrix, samples);
	cblas_daxpy(samples * numHidden, 1, resizedHiddenBiases, 1, hiddenMatrix, 1);
	std::transform(hiddenMatrix, hiddenMatrix + samples*numHidden, hiddenMatrix, [](double x) { return Utility::randFromBinomial(Utility::sigmoidFunction(x)); });
}

void RestrictedBoltzmannMachine::SampleFromVisibleGivenHidden(double* visibleMatrix, const double* hiddenMatrix, const double* resizedVisibleBiases, const int samples) const {
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, samples, numVisible, numHidden, 1, hiddenMatrix, samples, weightsMatrix, numVisible, 0, visibleMatrix, samples);
    cblas_daxpy(samples * numVisible, 1, resizedVisibleBiases, 1, visibleMatrix, 1);
	std::transform(visibleMatrix, visibleMatrix + samples*numVisible, visibleMatrix, [](double x) { return Utility::randFromBinomial(Utility::sigmoidFunction(x)); });
}

RestrictedBoltzmannMachine::RestrictedBoltzmannMachine(const int hidden, const int visible) {
    numVisible = visible;
    numHidden = hidden;
    weightsMatrix = new double[hidden * visible]();
    visibleBiases = new double[visible]();
    hiddenBiases = new double[hidden]();
    
	////For test run
	//for (int i = 0; i < hidden*visible; i++)
	//	weightsMatrix[i] = i+1;
	//std::fill_n(visibleBiases, visible, 1);
	//std::fill_n(hiddenBiases, hidden, 1);

	//For final run
	Utility::randVector(weightsMatrix, hidden * visible);
	Utility::randVector(visibleBiases, visible);
	Utility::randVector(hiddenBiases, hidden);
}

RestrictedBoltzmannMachine::~RestrictedBoltzmannMachine() {
    delete[] weightsMatrix;
    delete[] visibleBiases;
    delete[] hiddenBiases;
}


Utility::Gradients RestrictedBoltzmannMachine::ContrastiveDivergence(double* const data, const int samples, const int steps) const {
	// Memory allocation
	double* resizedHiddenBiases = new double[samples * numHidden];
	double* resizedVisibleBiases = new double[samples * numVisible];

	double* sampleVisiblePositive = new double[samples * numVisible];
	double* sampleHiddenPositive = new double[samples * numHidden];
	double* sampleVisibleNegative = new double[samples * numVisible];
	double* sampleHiddenNegative = new double[samples * numHidden];

	// Sampling
	for (int i = 0; i < samples; i++) {
		cblas_dcopy(numHidden, hiddenBiases, 1, resizedHiddenBiases + i, samples);
		cblas_dcopy(numVisible, visibleBiases, 1, resizedVisibleBiases + i, samples);
		cblas_dcopy(numVisible, data + i*numVisible, 1, sampleVisiblePositive + i, samples);
	}

	SampleFromHiddenGivenVisible(sampleHiddenPositive, sampleVisiblePositive, resizedHiddenBiases, samples);
	for (int i = 0; i < steps; i++) {
		SampleFromVisibleGivenHidden(sampleVisibleNegative, i == 0 ? sampleHiddenPositive : sampleHiddenNegative, resizedVisibleBiases, samples);
		SampleFromHiddenGivenVisible(sampleHiddenNegative, sampleVisibleNegative, resizedHiddenBiases, samples);
	}

	// Memory deallocation
	delete[] resizedHiddenBiases;
	delete[] resizedVisibleBiases;

	// Results
	return Utility::Gradients(sampleVisiblePositive, sampleHiddenPositive, sampleVisibleNegative, sampleHiddenNegative);
}

Utility::Deltas RestrictedBoltzmannMachine::GetDeltaWeights(const Utility::Gradients& gradients, const int samples, const double learningRate) const {
	// Memory allocation
	double* deltaWeightsMatrix1 = new double[numHidden * numVisible]();
    double* deltaWeightsMatrix2 = new double[numHidden * numVisible]();
    double* deltaVisibleBiases = new double[numVisible]();
    double* deltaHiddenBiases = new double[numHidden]();

	// Weights
    for(int i = 0; i < samples; i++) {
		double* v = gradients.visiblePositiveGradients + i;
		double* h = gradients.hiddenPositiveGradients + i;
		double* vprim = gradients.visibleNegativeGradients + i;
		double* hprim = gradients.hiddenNegativeGradients + i;

		cblas_dger(CblasColMajor, numVisible, numHidden, learningRate / samples, v, samples, h, samples, deltaWeightsMatrix1, numVisible);
		cblas_dger(CblasColMajor, numVisible, numHidden, learningRate / samples, vprim, samples, hprim, samples, deltaWeightsMatrix2, numVisible);
    }

	cblas_daxpy(numHidden * numVisible, -1, deltaWeightsMatrix2, 1, deltaWeightsMatrix1, 1);
	delete[] deltaWeightsMatrix2;

	// Additional memory allocation for biases
	double* ones = new double[samples]();
	std::fill_n(ones, samples, 1);

	// Visible biases
	cblas_daxpy(samples * numVisible, -1, gradients.visibleNegativeGradients, 1, gradients.visiblePositiveGradients, 1);
	cblas_dgemv(CblasColMajor, CblasTrans, samples, numVisible, learningRate / samples, gradients.visibleNegativeGradients, samples, ones, 1, 0, deltaVisibleBiases, 1);
	
	// Hidden biases
	cblas_daxpy(samples * numHidden, -1, gradients.hiddenNegativeGradients, 1, gradients.hiddenPositiveGradients, 1);
	cblas_dgemv(CblasColMajor, CblasTrans, samples, numHidden, learningRate / samples, gradients.hiddenPositiveGradients, samples, ones, 1, 0, deltaHiddenBiases, 1);
	
	// Additional memory deallocation for biases
	delete[] ones;

	// Results
	return Utility::Deltas(deltaWeightsMatrix1, deltaVisibleBiases, deltaHiddenBiases);
}


Utility::Deltas RestrictedBoltzmannMachine::TrainingIteration(double* const data, const int samples, const double learningRate, const int cdSteps) const {
	Utility::Gradients g = ContrastiveDivergence(data, samples, cdSteps);
	return GetDeltaWeights(g, samples, learningRate);
}

void RestrictedBoltzmannMachine::TrainNetwork(double* const data, const int samples, const double learningRate, const int miniBatchSize, const int epochs, const int cdSteps) { 
	int numMiniBatches = static_cast<int>(ceil(samples * 1.0 / miniBatchSize));
	for (int i = 0; i < epochs; i++) {
		//TODO: MPI (do each iteration in a distributed environment; after distributed calculation cumulate results (synchronize) and perform change weights sequential)
		for (int j = 0; j < numMiniBatches; j++) {
			int currTranslation = j*miniBatchSize;
			int nextTranslation = (j + 1)*miniBatchSize;
			int currMiniBatch = nextTranslation > samples ? samples - currTranslation : miniBatchSize;

			Utility::Deltas d = TrainingIteration(data + currTranslation*numVisible, currMiniBatch, learningRate, cdSteps);

			//Change weights
			cblas_daxpy(numHidden * numVisible, 1, d.deltaWeightsMatrix, 1, weightsMatrix, 1);
			cblas_daxpy(numHidden, 1, d.deltaHiddenBiases, 1, hiddenBiases, 1);
			cblas_daxpy(numVisible, 1, d.deltaVisibleBiases, 1, visibleBiases, 1);
		}	
	}
}

std::string RestrictedBoltzmannMachine::GetWeights() const {
	std::stringstream ss;
	for (int i = 0; i < numVisible; i++) {
		for(int j = 0; j < numHidden; j++)
			ss << std::setprecision(2) << std::fixed << (weightsMatrix[j + numHidden*i] >= 0.0 ? "+" : "") << weightsMatrix[j + numHidden*i] << " ";
		ss << std::endl;
	}

	return ss.str();
}

void RestrictedBoltzmannMachine::SaveWeights(const std::string filename) const {
	std::ofstream out(filename);
	out << GetWeights();
	out.close();
}