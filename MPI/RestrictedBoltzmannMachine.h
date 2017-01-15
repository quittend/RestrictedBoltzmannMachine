#ifndef RBM_H
#define RBM_H

#include <math.h>
#include <string>
#include "Utility.h"

class RestrictedBoltzmannMachine {
    private:
        std::shared_ptr<double> weightsMatrix;
        std::shared_ptr<double> visibleBiases;
        std::shared_ptr<double> hiddenBiases;
        int numVisible;
        int numHidden;

        void SampleFromHiddenGivenVisible(const std::shared_ptr<double>& hiddenMatrix, const std::shared_ptr<const double>& visibleMatrix, const std::shared_ptr<const double>& resizedHiddenBiases, const int samples) const;
        void SampleFromVisibleGivenHidden(const std::shared_ptr<double>& visibleMatrix, const std::shared_ptr<const double>& hiddenMatrix, const std::shared_ptr<const double>& resizedVisibleBiases, const int samples) const;
        Utility::Gradients ContrastiveDivergence(double* const visibleMatrix, const int samples, const int cdSteps) const;
        Utility::Deltas GetDeltaWeights(const Utility::Gradients& gradients, const int samples, const double learningRate) const;
        Utility::Deltas TrainingIteration(double* const data, const int samples, const double learningRate, const int cdSteps) const;

    public:
        RestrictedBoltzmannMachine(const int numHidden, const int numVisible);

        void TrainBatch(const std::shared_ptr<double>& data, const int samples, const double learningRate, const int miniBatchSize, const int epochs, const int cdSteps);
        std::shared_ptr<double> getWeights() const {return weightsMatrix;};
};

#endif
