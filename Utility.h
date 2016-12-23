#ifndef UTILS_H
#define UTILS_H

#define INIT_WEIGHT_MIN -0.1
#define INIT_WEIGHT_MAX 0.1

namespace Utility {
	double randDouble(const double min, const double max);
	void randVector(double* vector, const int size);
	int randFromBinomial(const double p);
	double sigmoidFunction(const double value);
	double* readMnistImagesBinarized(std::string fullPath, int count);

	struct Gradients {
			double* visiblePositiveGradients;
			double* hiddenPositiveGradients;
			double* visibleNegativeGradients;
			double* hiddenNegativeGradients;
			Gradients(double* vp, double* hp, double* vn, double* hn);
			~Gradients();
	};

	struct Deltas {
			double* deltaWeightsMatrix;
			double* deltaVisibleBiases;
			double* deltaHiddenBiases;
			Deltas(double* w, double* v, double* h);
			~Deltas();
	};
}

#endif
