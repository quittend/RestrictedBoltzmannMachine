#include <sstream> //stringstream
#include <string> //string
#include <fstream> //ofstream
#include <cmath> //ceil
#include <algorithm> //transform
extern "C" {
#include <cblas.h>
}
#include "Utility.h"
#include "RestrictedBoltzmannMachine.h"

#include <iostream>

void RestrictedBoltzmannMachine::SampleFromHiddenGivenVisible(const std::shared_ptr<double>& hiddenMatrix, const std::shared_ptr<const double>& visibleMatrix, const std::shared_ptr<const double>& resizedHiddenBiases, const int samples) const {
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, samples, numHidden, numVisible, 1, visibleMatrix.get(), samples, weightsMatrix.get(), numVisible, 0, hiddenMatrix.get(), samples);
    cblas_daxpy(samples * numHidden, 1, resizedHiddenBiases.get(), 1, hiddenMatrix.get(), 1);
    std::transform(hiddenMatrix.get(), hiddenMatrix.get() + samples*numHidden, hiddenMatrix.get(), [](double x) { return Utility::randFromBinomial(Utility::sigmoidFunction(x)); });
}

void RestrictedBoltzmannMachine::SampleFromVisibleGivenHidden(const std::shared_ptr<double>& visibleMatrix, const std::shared_ptr<const double>& hiddenMatrix, const std::shared_ptr<const double>& resizedVisibleBiases, const int samples) const {
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, samples, numVisible, numHidden, 1, hiddenMatrix.get(), samples, weightsMatrix.get(), numVisible, 0, visibleMatrix.get(), samples);
    cblas_daxpy(samples * numVisible, 1, resizedVisibleBiases.get(), 1, visibleMatrix.get(), 1);
    std::transform(visibleMatrix.get(), visibleMatrix.get() + samples*numVisible, visibleMatrix.get(), [](double x) { return Utility::randFromBinomial(Utility::sigmoidFunction(x)); });
}

RestrictedBoltzmannMachine::RestrictedBoltzmannMachine(const int hidden, const int visible) {
    numVisible = visible;
    numHidden = hidden;
    weightsMatrix = std::shared_ptr<double>(new double[hidden * visible], std::default_delete<double[]>());
    visibleBiases = std::shared_ptr<double>(new double[visible], std::default_delete<double[]>());
    hiddenBiases = std::shared_ptr<double>(new double[hidden], std::default_delete<double[]>());

    //For test run
    for (int i = 0; i < hidden*visible; i++)
        weightsMatrix.get()[i] = i+1;
    std::fill_n(visibleBiases.get(), visible, 1);
    std::fill_n(hiddenBiases.get(), hidden, 1);

    // //For final run
    // Utility::randVector(weightsMatrix, hidden * visible);
    // Utility::randVector(visibleBiases, visible);
    // Utility::randVector(hiddenBiases, hidden);
}


Utility::Gradients RestrictedBoltzmannMachine::ContrastiveDivergence(double* const data, const int samples, const int steps) const {
    // Memory allocation
    std::shared_ptr<double> resizedHiddenBiases = std::shared_ptr<double>(new double[samples * numHidden], std::default_delete<double[]>());
    std::shared_ptr<double> resizedVisibleBiases = std::shared_ptr<double>(new double[samples * numVisible], std::default_delete<double[]>());

    std::shared_ptr<double> sampleVisiblePositive = std::shared_ptr<double>(new double[samples * numVisible], std::default_delete<double[]>());
    std::shared_ptr<double> sampleHiddenPositive = std::shared_ptr<double>(new double[samples * numHidden], std::default_delete<double[]>());
    std::shared_ptr<double> sampleVisibleNegative = std::shared_ptr<double>(new double[samples * numVisible], std::default_delete<double[]>());
    std::shared_ptr<double> sampleHiddenNegative = std::shared_ptr<double>(new double[samples * numHidden], std::default_delete<double[]>());

    // Sampling
    for (int i = 0; i < samples; i++) {
        cblas_dcopy(numHidden, hiddenBiases.get(), 1, resizedHiddenBiases.get() + i, samples);
        cblas_dcopy(numVisible, visibleBiases.get(), 1, resizedVisibleBiases.get() + i, samples);
        cblas_dcopy(numVisible, data + i*numVisible, 1, sampleVisiblePositive.get() + i, samples);
    }

    SampleFromHiddenGivenVisible(sampleHiddenPositive, sampleVisiblePositive, resizedHiddenBiases, samples);
    for (int i = 0; i < steps; i++) {
        SampleFromVisibleGivenHidden(sampleVisibleNegative, i == 0 ? sampleHiddenPositive : sampleHiddenNegative, resizedVisibleBiases, samples);
        SampleFromHiddenGivenVisible(sampleHiddenNegative, sampleVisibleNegative, resizedHiddenBiases, samples);
    }

    // Results
    return Utility::Gradients(sampleVisiblePositive, sampleHiddenPositive, sampleVisibleNegative, sampleHiddenNegative);
}

Utility::Deltas RestrictedBoltzmannMachine::GetDeltaWeights(const Utility::Gradients& gradients, const int samples, const double learningRate) const {
    // Memory allocation
    std::shared_ptr<double> deltaWeightsMatrix1 = std::shared_ptr<double>(new double[numHidden * numVisible](), std::default_delete<double[]>());
    std::shared_ptr<double> deltaWeightsMatrix2 = std::shared_ptr<double>(new double[numHidden * numVisible](), std::default_delete<double[]>());
    std::shared_ptr<double> deltaVisibleBiases = std::shared_ptr<double>(new double[numVisible](), std::default_delete<double[]>());
    std::shared_ptr<double> deltaHiddenBiases = std::shared_ptr<double>(new double[numHidden](), std::default_delete<double[]>());

    // Weights
    for(int i = 0; i < samples; i++) {
        double* v = gradients.visiblePositiveGradients.get() + i;
        double* h = gradients.hiddenPositiveGradients.get() + i;
        double* vprim = gradients.visibleNegativeGradients.get() + i;
        double* hprim = gradients.hiddenNegativeGradients.get() + i;

        cblas_dger(CblasColMajor, numVisible, numHidden, learningRate / samples, v, samples, h, samples, deltaWeightsMatrix1.get(), numVisible);
        cblas_dger(CblasColMajor, numVisible, numHidden, learningRate / samples, vprim, samples, hprim, samples, deltaWeightsMatrix2.get(), numVisible);
    }

    cblas_daxpy(numHidden * numVisible, -1, deltaWeightsMatrix2.get(), 1, deltaWeightsMatrix1.get(), 1);

    // Additional memory allocation for biases
    std::shared_ptr<double> ones = std::shared_ptr<double>(new double[samples](), std::default_delete<double[]>());
    std::fill_n(ones.get(), samples, 1);

    // Visible biases
    cblas_daxpy(samples * numVisible, -1, gradients.visibleNegativeGradients.get(), 1, gradients.visiblePositiveGradients.get(), 1);
    cblas_dgemv(CblasColMajor, CblasTrans, samples, numVisible, learningRate / samples, gradients.visibleNegativeGradients.get(), samples, ones.get(), 1, 0, deltaVisibleBiases.get(), 1);

    // Hidden biases
    cblas_daxpy(samples * numHidden, -1, gradients.hiddenNegativeGradients.get(), 1, gradients.hiddenPositiveGradients.get(), 1);
    cblas_dgemv(CblasColMajor, CblasTrans, samples, numHidden, learningRate / samples, gradients.hiddenPositiveGradients.get(), samples, ones.get(), 1, 0, deltaHiddenBiases.get(), 1);

    // Results
    return Utility::Deltas(deltaWeightsMatrix1, deltaVisibleBiases, deltaHiddenBiases);
}


Utility::Deltas RestrictedBoltzmannMachine::TrainingIteration(double* const data, const int samples, const double learningRate, const int cdSteps) const {
    Utility::Gradients g = ContrastiveDivergence(data, samples, cdSteps);
    return GetDeltaWeights(g, samples, learningRate);
}

void RestrictedBoltzmannMachine::TrainBatch(const std::shared_ptr<double>& data, const int samples, const double learningRate, const int miniBatchSize, const int epochs, const int cdSteps) {
    int numMiniBatches = static_cast<int>(ceil(samples * 1.0 / miniBatchSize));
    for (int i = 0; i < epochs; i++) {
        for (int j = 0; j < numMiniBatches; j++) {
            int currTranslation = j*miniBatchSize;
            int nextTranslation = (j + 1)*miniBatchSize;
            int currMiniBatch = nextTranslation > samples ? samples - currTranslation : miniBatchSize;

            Utility::Deltas d = TrainingIteration(data.get() + currTranslation*numVisible, currMiniBatch, learningRate, cdSteps);

            Utility::exchangeDeltas(d, numVisible, numHidden);

            //Change weights
            cblas_daxpy(numHidden * numVisible, 1, d.deltaWeightsMatrix.get(), 1, weightsMatrix.get(), 1);
            cblas_daxpy(numHidden, 1, d.deltaHiddenBiases.get(), 1, hiddenBiases.get(), 1);
            cblas_daxpy(numVisible, 1, d.deltaVisibleBiases.get(), 1, visibleBiases.get(), 1);
        }
    }
}
