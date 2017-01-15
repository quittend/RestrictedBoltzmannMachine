#include <random>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <iomanip> //setprecision
#include "Utility.h"
#include <sstream>
#include <mpi.h>

#include <iostream>

namespace Utility {

    const unsigned int SND_BATCH = 0;
    const unsigned int SND_WEIGHTS = 1;
    const unsigned int SND_VISIBLE_BIASES = 2;
    const unsigned int SND_HIDDEN_BIASES = 3;
    const unsigned int SND_RESULT = 4;

    void saveWeights(const std::string& filename, const std::shared_ptr<double>& weights, const int visible, const int hidden) {
        std::ofstream out(filename, std::ios::out);
        for(int i = 0; i < visible; i++) {
            for(int j = 0; j < hidden; j++) {
                out << std::setprecision(2) << std::fixed << (weights.get()[j + hidden*i] >= 0.0 ? "+" : "") << weights.get()[j + hidden*i] << " ";
            }
            out << std::endl;
        }
        out.close();
    }


    double randDouble(const double min, const double max) {
        std::random_device rd;
        std::mt19937 rng(rd());
        std::uniform_real_distribution<double> uni(min, max);
        return uni(rng);
    }

    void randVector(std::shared_ptr<double>& vector, const int size) {
        std::transform(vector.get(), vector.get() + size + 1, vector.get(), [](double x) { return randDouble(INIT_WEIGHT_MIN, INIT_WEIGHT_MAX); });
    }

    int randFromBinomial(const double p) {
        std::default_random_engine generator;
        std::binomial_distribution<int> bi(1, p);
        return bi(generator);
    }

    std::shared_ptr<double> readMnistImagesBinarized(std::string fullPath, int count) {
        auto reverseInt = [](int i) {
            unsigned char c1, c2, c3, c4;
            c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
            return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
        };

        std::ifstream file(fullPath, std::ios::binary);

        if (file.is_open()) {
            int magicNumber = 0, nRows = 0, nCols = 0;

            file.read((char *)&magicNumber, sizeof(magicNumber));
            magicNumber = reverseInt(magicNumber);

            if (magicNumber != 2051) throw std::runtime_error("Invalid MNIST image file!");

            int numberOfImages;
            file.read((char *)&numberOfImages, sizeof(numberOfImages)), numberOfImages = reverseInt(numberOfImages);
            file.read((char *)&nRows, sizeof(nRows)), nRows = reverseInt(nRows);
            file.read((char *)&nCols, sizeof(nCols)), nCols = reverseInt(nCols);

            int imageSize = nRows * nCols;
            numberOfImages = count;

            std::shared_ptr<double> dataset = std::shared_ptr<double>(new double[numberOfImages * imageSize], std::default_delete<double[]>());
            unsigned char* temp = new unsigned char[numberOfImages * imageSize];
            for (int i = 0; i < numberOfImages; i++)
                file.read((char *)(temp + i*(imageSize)), imageSize);

            for (int i = 0; i < numberOfImages * imageSize; i++)
                dataset.get()[i] = temp[i] > 30 ? 1.0 : 0.0;
            delete[] temp;

            return dataset;
        }
        else
            throw std::runtime_error("Cannot open file `" + fullPath + "`!");
    }


    double sigmoidFunction(const double value) { return 1 / (1 + exp(-value)); }

    Gradients::Gradients(std::shared_ptr<double> vp, std::shared_ptr<double> hp, std::shared_ptr<double> vn, std::shared_ptr<double> hn) :
        visiblePositiveGradients(vp), hiddenPositiveGradients(hp), visibleNegativeGradients(vn), hiddenNegativeGradients(hn) {}


    Deltas::Deltas(std::shared_ptr<double> w, std::shared_ptr<double> v, std::shared_ptr<double> h) :
        deltaWeightsMatrix(w), deltaVisibleBiases(v), deltaHiddenBiases(h) {}


    void mergeDeltas(Deltas& d1, const Deltas& d2, const int visible, const int hidden) {
        for(int i=0; i<visible*hidden; ++i) {
            d1.deltaWeightsMatrix.get()[i] += d2.deltaWeightsMatrix.get()[i];
        }
        for(int i=0; i<visible; ++i) {
            d1.deltaVisibleBiases.get()[i] += d2.deltaVisibleBiases.get()[i];
        }
        for(int i=0; i<hidden; ++i) {
            d1.deltaHiddenBiases.get()[i] += d2.deltaHiddenBiases.get()[i];
        }
    }


    void sendDeltas(const Deltas& d, const int peerNo, const int visible, const int hidden) {
        MPI_Send(d.deltaWeightsMatrix.get(), visible*hidden, MPI_DOUBLE, peerNo, SND_WEIGHTS, MPI_COMM_WORLD);
        MPI_Send(d.deltaVisibleBiases.get(), visible, MPI_DOUBLE, peerNo, SND_VISIBLE_BIASES, MPI_COMM_WORLD);
        MPI_Send(d.deltaHiddenBiases.get(), hidden, MPI_DOUBLE, peerNo, SND_HIDDEN_BIASES, MPI_COMM_WORLD);
    }


    Deltas recvDeltas(const int peerNo, const int visible, const int hidden) {
        Deltas d;
        d.deltaWeightsMatrix = std::shared_ptr<double>(new double[visible*hidden], std::default_delete<double[]>());
        d.deltaVisibleBiases = std::shared_ptr<double>(new double[visible], std::default_delete<double[]>());
        d.deltaHiddenBiases = std::shared_ptr<double>(new double[hidden], std::default_delete<double[]>());

        MPI_Recv(d.deltaWeightsMatrix.get(), hidden*visible, MPI_DOUBLE, peerNo, SND_WEIGHTS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(d.deltaVisibleBiases.get(), visible, MPI_DOUBLE, peerNo, SND_VISIBLE_BIASES, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(d.deltaHiddenBiases.get(), hidden, MPI_DOUBLE, peerNo, SND_HIDDEN_BIASES, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        return d;
    }


    void mergeWithPeer(Deltas& d, const int peerNo, const int visible, const int hidden) {
        int world_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        if(world_rank < peerNo) {
            sendDeltas(d, peerNo, visible, hidden);
            d = recvDeltas(peerNo, visible, hidden);
        } else {
            Deltas peerDeltas = recvDeltas(peerNo, visible, hidden);
            mergeDeltas(d, peerDeltas, visible, hidden);
            sendDeltas(d, peerNo, visible, hidden);
        }
    }


    void exchangeDeltas(Deltas& d, const int visible, const int hidden) {
        int world_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        int world_size;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        LOG("W" << world_rank << " exchanging deltas\n");
        int j=1; // jump length
        for(int n=0; n<ceil(log2(world_size)); ++n) {
            if((world_rank-1)%(2*j)-j < 0) {
                if(world_rank+j < world_size) {
                    LOG("W" << world_rank << " merge deltas with peer " << world_rank+j << "\n");
                    mergeWithPeer(d, world_rank+j, visible, hidden);
                }
            } else {
                LOG("W" << world_rank << " merge deltas with peer " << world_rank-j << "\n");
                mergeWithPeer(d, world_rank-j, visible, hidden);
            }
            j*=2;
        }
        if((world_size-1)%(j/2) != 0) { // wyslij delty do peerow ktorzy nie brali udzialu w ostatniej wymianie
            int firstPeer = (world_size-1)%(j/2)+1;
            int lastPeer = j/2;
            if(world_rank == 1) {
                for(int s=firstPeer; s<=lastPeer; ++s) {
                    LOG("W" << world_rank << " send deltas to peer " << s << "\n");
                    sendDeltas(d, s, visible, hidden);
                }
            } else if(world_rank >= firstPeer && world_rank <= lastPeer) {
                d = recvDeltas(1, visible, hidden);
            }
        }
    }
}
