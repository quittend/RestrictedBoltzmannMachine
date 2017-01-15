#ifndef UTILS_H
#define UTILS_H

#include <memory>

#define INIT_WEIGHT_MIN -0.1
#define INIT_WEIGHT_MAX 0.1

namespace Utility {

    extern const unsigned int SND_BATCH;
    extern const unsigned int SND_WEIGHTS;
    extern const unsigned int SND_VISIBLE_BIASES;
    extern const unsigned int SND_HIDDEN_BIASES;
    extern const unsigned int SND_RESULT;

    double randDouble(const double min, const double max);
    void randVector(std::shared_ptr<double>& vector, const int size);
    int randFromBinomial(const double p);
    double sigmoidFunction(const double value);
    std::shared_ptr<double> readMnistImagesBinarized(std::string fullPath, int count);

    // void saveWeights(const std::string& filename, const std::shared_ptr<double>& weights, const int visible, const int hidden);
    void saveWeights(const std::string& filename, const std::shared_ptr<double>& weights, const int visible, const int hidden);

    struct Gradients {
        std::shared_ptr<double> visiblePositiveGradients;
        std::shared_ptr<double> hiddenPositiveGradients;
        std::shared_ptr<double> visibleNegativeGradients;
        std::shared_ptr<double> hiddenNegativeGradients;
        Gradients(std::shared_ptr<double> vp, std::shared_ptr<double> hp, std::shared_ptr<double> vn, std::shared_ptr<double> hn);
    };

    struct Deltas {
        std::shared_ptr<double> deltaWeightsMatrix;
        std::shared_ptr<double> deltaVisibleBiases;
        std::shared_ptr<double> deltaHiddenBiases;
        Deltas() {};
        Deltas(std::shared_ptr<double> w, std::shared_ptr<double> v, std::shared_ptr<double> h);
    };

    void mergeDeltas(Deltas& d1, const Deltas& d2, const int visible, const int hidden);

    void sendDeltas(const Deltas& d, const int peerNo, const int visible, const int hidden);

    Deltas recvDeltas(const int peerNo, const int visible, const int hidden);

    void mergeWithPeer(Deltas& d, const int peerNo, const int visible, const int hidden);

    void exchangeDeltas(Deltas& d, const int visible, const int hidden);
}

#endif
