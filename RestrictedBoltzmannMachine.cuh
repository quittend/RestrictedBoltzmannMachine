#ifndef RBM_H
#define RBM_H

#include <math.h>
#include <string>
#include <cublas_v2.h>
#include "Utility.cuh"

class RestrictedBoltzmannMachine {
	private:
		double* weightsMatrix;
		double* visibleBiases;
		double* hiddenBiases;
		double* weightsMatrixDevice;
		double* visibleBiasesDevice;
		double* hiddenBiasesDevice;

		int numVisible;
		int numHidden;
		cublasHandle_t defaultHandle;

	public:
		RestrictedBoltzmannMachine(const int numHidden, const int numVisible);
		~RestrictedBoltzmannMachine();

		void SampleFromHiddenGivenVisible(double* const hiddenMatrixDevice, const double* const visibleMatrixDevice, const double* const resizedHiddenBiasesDevice, const int samples) const;
		void SampleFromVisibleGivenHidden(double* const visibleMatrixDevice, const double* const hiddenMatrixDevice, const double* const resizedVisibleBiasesDevice, const int samples) const;
		Utility::Gradients ContrastiveDivergence(double* const dataDevice, const int samples, const int cdSteps) const;
		Utility::Deltas GetDeltaWeights(Utility::Gradients& gradients, const int samples, const double learningRate) const;
		Utility::Deltas TrainingIteration(double* const dataDevice, const int samples, const double learningRate, const int cdSteps) const;

		void TrainNetwork(double* const dataDevice, const int samples, const double learningRate, const int miniBatchSize, const int epochs, const int cdSteps);
		std::string GetWeights() const;
		void SaveWeights(const std::string filename) const;

};

#endif