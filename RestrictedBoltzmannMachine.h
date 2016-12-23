#ifndef RBM_H
#define RBM_H

#include <math.h>
#include <string>
#include "Utility.h"

class RestrictedBoltzmannMachine {
	private:
		double* weightsMatrix;
        double* visibleBiases;
        double* hiddenBiases;
		int numVisible;
		int numHidden;

		void SampleFromHiddenGivenVisible(double* const hiddenMatrix, const double* const visibleMatrix, const double* const resizedHiddenBiases, const int samples) const;
		void SampleFromVisibleGivenHidden(double* const visibleMatrix, const double* const hiddenMatrix, const double* const resizedVisibleBiases, const int samples) const;
		Utility::Gradients ContrastiveDivergence(double* const visibleMatrix, const int samples, const int cdSteps) const;
		Utility::Deltas GetDeltaWeights(const Utility::Gradients& gradients, const int samples, const double learningRate) const;
		Utility::Deltas TrainingIteration(double* const data, const int samples, const double learningRate, const int cdSteps) const;

	public:
        RestrictedBoltzmannMachine(const int numHidden, const int numVisible);
        ~RestrictedBoltzmannMachine();

        void TrainNetwork(double* const data, const int samples, const double learningRate, const int miniBatchSize, const int epochs, const int cdSteps);
        std::string GetWeights() const;
        void SaveWeights(const std::string filename) const;
};

#endif
