#include <sstream> //stringstream
#include <string> //string
#include <iomanip> //setprecision
#include <fstream> //ofstream
#include <cmath> //floor
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <thrust/transform.h>
#include <thrust/device_ptr.h>
#include "Utility.cuh"
#include "RestrictedBoltzmannMachine.cuh"


void RestrictedBoltzmannMachine::SampleFromHiddenGivenVisible(double* const hiddenMatrixDevice, const double* const visibleMatrixDevice, const double* resizedHiddenBiasesDevice, const int samples) const {
	static const double alpha = 1.0;
	static const double beta = 0.0;
	cublasDgemm(defaultHandle, CUBLAS_OP_N, CUBLAS_OP_N, samples, numHidden, numVisible, &alpha, visibleMatrixDevice, samples, weightsMatrixDevice, numVisible, &beta, hiddenMatrixDevice, samples);
	cublasDaxpy(defaultHandle, samples * numHidden, &alpha, resizedHiddenBiasesDevice, 1, hiddenMatrixDevice, 1);

	const auto itDevice = thrust::device_pointer_cast(hiddenMatrixDevice);
	thrust::transform(itDevice, itDevice + samples*numHidden, itDevice, [] __device__ (double x) { return Utility::randFromBinomial(Utility::sigmoidFunction(x)); });
}

void RestrictedBoltzmannMachine::SampleFromVisibleGivenHidden(double* const visibleMatrixDevice, const double* const hiddenMatrixDevice, const double* resizedVisibleBiasesDevice, const int samples) const {
	static const double alpha = 1.0;
	static const double beta = 0.0;
	cublasDgemm(defaultHandle, CUBLAS_OP_N, CUBLAS_OP_T, samples, numVisible, numHidden, &alpha, hiddenMatrixDevice, samples, weightsMatrixDevice, numVisible, &beta, visibleMatrixDevice, samples);
	cublasDaxpy(defaultHandle, samples * numVisible, &alpha, resizedVisibleBiasesDevice, 1, visibleMatrixDevice, 1);
	
	const auto itDevice = thrust::device_pointer_cast(visibleMatrixDevice);
	thrust::transform(itDevice, itDevice + samples*numVisible, itDevice, [] __device__ (double x) { return Utility::randFromBinomial(Utility::sigmoidFunction(x)); });
}

Utility::Gradients RestrictedBoltzmannMachine::ContrastiveDivergence(double* const dataDevice, const int samples, const int cdSteps) const
{
	// Memory allocation
	double* resizedHiddenBiasesDevice; cudaMalloc(&resizedHiddenBiasesDevice, sizeof(double)*samples*numHidden);
	double* resizedVisibleBiasesDevice;  cudaMalloc(&resizedVisibleBiasesDevice, sizeof(double)*samples*numVisible);

	double* sampleVisiblePositiveDevice; cudaMalloc(&sampleVisiblePositiveDevice, sizeof(double) * samples * numVisible);
	double* sampleHiddenPositiveDevice;	cudaMalloc(&sampleHiddenPositiveDevice, sizeof(double) * samples * numHidden);
	double* sampleVisibleNegativeDevice; cudaMalloc(&sampleVisibleNegativeDevice, sizeof(double) * samples * numVisible);
	double* sampleHiddenNegativeDevice;	cudaMalloc(&sampleHiddenNegativeDevice, sizeof(double) * samples * numHidden);

	cudaStream_t* streams = new cudaStream_t[samples];
	cublasHandle_t* handles = new cublasHandle_t[samples];
	for (int i = 0; i < samples; i++) {
		cublasCreate(&handles[i]);
		cudaStreamCreate(&streams[i]);
		cublasSetStream(handles[i], streams[i]);
	}

	for (int i = 0; i < samples; i++) {
		cublasDcopy(handles[i], numHidden, hiddenBiasesDevice, 1, resizedHiddenBiasesDevice + i, samples);
		cublasDcopy(handles[i], numVisible, visibleBiasesDevice, 1, resizedVisibleBiasesDevice + i, samples);
		cublasDcopy(defaultHandle, numVisible, dataDevice + i*numVisible, 1, sampleVisiblePositiveDevice + i, samples);
		cudaStreamDestroy(streams[i]);
	}

	// Sampling
	SampleFromHiddenGivenVisible(sampleHiddenPositiveDevice, sampleVisiblePositiveDevice, resizedHiddenBiasesDevice, samples);
	for (int i = 0; i < cdSteps; i++) {
		SampleFromVisibleGivenHidden(sampleVisibleNegativeDevice, i == 0 ? sampleHiddenPositiveDevice : sampleHiddenNegativeDevice, resizedVisibleBiasesDevice, samples);
		SampleFromHiddenGivenVisible(sampleHiddenNegativeDevice, sampleVisibleNegativeDevice, resizedHiddenBiasesDevice, samples);
	}

	// Memory deallocation
	cudaFree(resizedHiddenBiasesDevice);
	cudaFree(resizedVisibleBiasesDevice);
	delete[] handles;
	delete[] streams;

	// Results
	return Utility::Gradients(sampleVisiblePositiveDevice, sampleHiddenPositiveDevice, sampleVisibleNegativeDevice, sampleHiddenNegativeDevice);
}

RestrictedBoltzmannMachine::RestrictedBoltzmannMachine(const int hidden, const int visible) {
	numVisible = visible;
	numHidden = hidden;

	// Allocate memory
	weightsMatrix = new double[hidden * visible]();
	visibleBiases = new double[visible]();
	hiddenBiases = new double[hidden]();
	cudaMalloc(&weightsMatrixDevice, sizeof(double)*visible*hidden);
	cudaMalloc(&hiddenBiasesDevice, sizeof(double)*hidden);
	cudaMalloc(&visibleBiasesDevice, sizeof(double)*visible);

	//For test run
	//for (int i = 0; i < hidden*visible; i++)
	//	weightsMatrix[i] = i+1;
	//std::fill_n(visibleBiases, visible, 1);
	//std::fill_n(hiddenBiases, hidden, 1);

	//For final run
	Utility::randVector(weightsMatrix, hidden * visible);
	Utility::randVector(visibleBiases, visible);
	Utility::randVector(hiddenBiases, hidden);

	// Copy data to a device
	cudaMemcpy(weightsMatrixDevice, weightsMatrix, sizeof(double)*visible*hidden, cudaMemcpyHostToDevice);
	cudaMemcpy(visibleBiasesDevice, visibleBiases, sizeof(double)*visible, cudaMemcpyHostToDevice);
	cudaMemcpy(hiddenBiasesDevice, hiddenBiases, sizeof(double)*hidden, cudaMemcpyHostToDevice);

	// Handle creation
	cublasCreate(&defaultHandle);
}

RestrictedBoltzmannMachine::~RestrictedBoltzmannMachine() {
	delete[] weightsMatrix;
	delete[] visibleBiases;
	delete[] hiddenBiases;
	cudaFree(weightsMatrixDevice);
	cudaFree(visibleBiasesDevice);
	cudaFree(hiddenBiasesDevice);
	cublasDestroy(defaultHandle);
}


Utility::Deltas RestrictedBoltzmannMachine::GetDeltaWeights(Utility::Gradients& gradients, const int samples, const double learningRate) const {
	// Memory allocation
	double* deltaWeightsMatrix1Device; cudaMalloc(&deltaWeightsMatrix1Device, sizeof(double)*numHidden*numVisible);
	double* deltaWeightsMatrix2Device; cudaMalloc(&deltaWeightsMatrix2Device, sizeof(double)*numHidden*numVisible);
	double* deltaVisibleBiasesDevice; cudaMalloc(&deltaVisibleBiasesDevice, sizeof(double)*numVisible);
	double* deltaHiddenBiasesDevice; cudaMalloc(&deltaHiddenBiasesDevice, sizeof(double)*numHidden);
	thrust::device_ptr<double> it = thrust::device_pointer_cast(deltaWeightsMatrix1Device); 
	thrust::fill_n(it, numHidden*numVisible, 0.0);
	it = thrust::device_pointer_cast(deltaWeightsMatrix2Device); thrust::fill_n(it, numHidden*numVisible, 0.0);
	it = thrust::device_pointer_cast(deltaVisibleBiasesDevice); thrust::fill_n(it, numVisible, 0.0);
	it = thrust::device_pointer_cast(deltaHiddenBiasesDevice); thrust::fill_n(it, numHidden, 0.0);

	cudaStream_t* streams = new cudaStream_t[samples];
	cublasHandle_t* handles = new cublasHandle_t[samples];
	for (int i = 0; i < samples; i++) {
		cublasCreate(&handles[i]);
		cudaStreamCreate(&streams[i]);
		cublasSetStream(handles[i], streams[i]);
	}

	// Weights (might be and issue, because Kosinski said so)
	const double alpha = learningRate / samples;
	for (int i = 0; i < samples; i++) {
		double* v = gradients.visiblePositiveGradientsDevice + i;
		double* h = gradients.hiddenPositiveGradientsDevice + i;
		double* vprim = gradients.visibleNegativeGradientsDevice + i;
		double* hprim = gradients.hiddenNegativeGradientsDevice + i;

		cublasDger(handles[i], numVisible, numHidden, &alpha, v, samples, h, samples, deltaWeightsMatrix1Device, numVisible);
		cublasDger(handles[i], numVisible, numHidden, &alpha, vprim, samples, hprim, samples, deltaWeightsMatrix2Device, numVisible);
		cudaStreamDestroy(streams[i]);
	}

	delete[] handles;
	delete[] streams;

	static const double minusOne = -1.0;
	cublasDaxpy(defaultHandle, numHidden * numVisible, &minusOne, deltaWeightsMatrix2Device, 1, deltaWeightsMatrix1Device, 1);
	cudaFree(deltaWeightsMatrix2Device);

	// Additional memory allocation for biases
	double* onesDevice; cudaMalloc(&onesDevice, sizeof(double)*samples);
	it = thrust::device_pointer_cast(onesDevice); thrust::fill_n(it, samples, 1.0);

	cudaStream_t streamsBiases[2];
	cublasHandle_t handlesBiases[2];

	for (int i = 0; i < 2; i++) {
		cublasCreate(&handlesBiases[i]);
		cudaStreamCreate(&streamsBiases[i]);
		cublasSetStream(handlesBiases[i], streamsBiases[i]);
	}

	static const double beta = 0.0;
	// Visible biases
	cublasDaxpy(handlesBiases[0], samples * numVisible, &minusOne, gradients.visibleNegativeGradientsDevice, 1, gradients.visiblePositiveGradientsDevice, 1);
	cublasDgemv(handlesBiases[0], CUBLAS_OP_T, samples, numVisible, &alpha, gradients.visiblePositiveGradientsDevice, samples, onesDevice, 1, &beta, deltaVisibleBiasesDevice, 1);
	
	// Hidden biases
	cublasDaxpy(handlesBiases[1], samples * numHidden, &minusOne, gradients.hiddenNegativeGradientsDevice, 1, gradients.hiddenPositiveGradientsDevice, 1);
	cublasDgemv(handlesBiases[1], CUBLAS_OP_T, samples, numHidden, &alpha, gradients.hiddenPositiveGradientsDevice, samples, onesDevice, 1, &beta, deltaHiddenBiasesDevice, 1);

	cudaStreamDestroy(streamsBiases[0]);
	cudaStreamDestroy(streamsBiases[1]);

	// Additional memory deallocation for biases
	cudaFree(onesDevice);

	// Results
	return Utility::Deltas(deltaWeightsMatrix1Device, deltaVisibleBiasesDevice, deltaHiddenBiasesDevice);
}

Utility::Deltas RestrictedBoltzmannMachine::TrainingIteration(double* const dataDevice, const int samples, const double learningRate, const int cdSteps) const {
	Utility::Gradients g = ContrastiveDivergence(dataDevice, samples, cdSteps);
	return GetDeltaWeights(g, samples, learningRate);
}

void RestrictedBoltzmannMachine::TrainNetwork(double* const dataDevice, const int samples, const double learningRate, const int miniBatchSize, const int epochs, const int cdSteps) {
	int numMiniBatches = static_cast<int>(ceil(samples * 1.0 / miniBatchSize));
	for (int i = 0; i < epochs; i++) {
		for (int j = 0; j < numMiniBatches; j++) {
			int currTranslation = j*miniBatchSize;
			int nextTranslation = (j + 1)*miniBatchSize;
			int currMiniBatch = nextTranslation > samples ? samples - currTranslation : miniBatchSize;

			Utility::Deltas d = TrainingIteration(dataDevice + currTranslation*numVisible, currMiniBatch, learningRate, cdSteps);

			static const double alpha = 1.0;
			cublasDaxpy(defaultHandle, numHidden * numVisible, &alpha, d.deltaWeightsMatrixDevice, 1, weightsMatrixDevice, 1);
			cublasDaxpy(defaultHandle, numHidden, &alpha, d.deltaHiddenBiasesDevice, 1, hiddenBiasesDevice, 1);
			cublasDaxpy(defaultHandle, numVisible, &alpha, d.deltaVisibleBiasesDevice, 1, visibleBiasesDevice, 1);
		}
	}

	cudaMemcpy(weightsMatrix, weightsMatrixDevice, sizeof(double)*numVisible*numHidden, cudaMemcpyDeviceToHost);
	cudaMemcpy(hiddenBiases, hiddenBiasesDevice, sizeof(double)*numHidden, cudaMemcpyDeviceToHost);
	cudaMemcpy(visibleBiases, visibleBiasesDevice, sizeof(double)*numVisible, cudaMemcpyDeviceToHost);
}

std::string RestrictedBoltzmannMachine::GetWeights() const {
	std::stringstream ss;
	for (int i = 0; i < numVisible; i++) {
		for (int j = 0; j < numHidden; j++)
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