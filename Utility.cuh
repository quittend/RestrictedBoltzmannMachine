#ifndef UTILS_H
#define UTILS_H

#include <host_defines.h>
#include <thrust/random/uniform_real_distribution.h>
#include <thrust/random/linear_congruential_engine.h>
#include <string>
#define INIT_WEIGHT_MIN -0.1
#define INIT_WEIGHT_MAX 0.1

namespace Utility {
	double randDouble(const double min, const double max);
	void randVector(double* vector, const int size);

	inline __device__ int randFromBinomial(double p) {
		thrust::minstd_rand rng;
		thrust::uniform_real_distribution<double> uniform(0, 1);
		return uniform(rng) > p ? 1 : 0;
	}
	inline __device__  double sigmoidFunction(double value) { return 1 / (1 + exp(-value)); }

	double* readMnistImagesBinarized(std::string fullPath, int count);

	struct Gradients {
		double* visiblePositiveGradientsDevice;
		double* hiddenPositiveGradientsDevice;
		double* visibleNegativeGradientsDevice;
		double* hiddenNegativeGradientsDevice;
		Gradients(double* vpDevice, double* hpDevice, double* vnDevice, double* hnDevice);
		~Gradients();
	};

	struct Deltas {
		double* deltaWeightsMatrixDevice;
		double* deltaVisibleBiasesDevice;
		double* deltaHiddenBiasesDevice;
		Deltas(double* wDevice, double* vDevice, double* hDevice);
		~Deltas();
	};
}

#endif