#include <random>
#include <cmath>
#include <algorithm>
#include <fstream>
#include "Utility.h"

double Utility::randDouble(const double min, const double max) {
	std::random_device rd;
	std::mt19937 rng(rd());
	std::uniform_real_distribution<double> uni(min, max);
	return uni(rng);
}

void Utility::randVector(double* vector, const int size) {
	std::transform(vector, vector + size + 1, vector, [](double x) { return randDouble(INIT_WEIGHT_MIN, INIT_WEIGHT_MAX); });
}

int Utility::randFromBinomial(const double p) {
	std::default_random_engine generator;
	std::binomial_distribution<int> bi(1, p);
	return bi(generator);
}

double* Utility::readMnistImagesBinarized(std::string fullPath, int count) {
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

		double* dataset = new double[numberOfImages * imageSize];
		unsigned char* temp = new unsigned char[numberOfImages * imageSize];
		for (int i = 0; i < numberOfImages; i++)
			file.read((char *)(temp + i*(imageSize)), imageSize);

		for (int i = 0; i < numberOfImages * imageSize; i++)
			dataset[i] = temp[i] > 30 ? 1.0 : 0.0;
		delete[] temp;

		return dataset;
	}
	else
		throw std::runtime_error("Cannot open file `" + fullPath + "`!");
}


double Utility::sigmoidFunction(const double value) { return 1 / (1 + exp(-value)); }

Utility::Gradients::Gradients(double* vp, double* hp, double* vn, double* hn) :
	visiblePositiveGradients(vp), hiddenPositiveGradients(hp), visibleNegativeGradients(vn), hiddenNegativeGradients(hn) {}

Utility::Gradients::~Gradients() {
	delete[] visiblePositiveGradients;
	delete[] hiddenPositiveGradients;
	delete[] visibleNegativeGradients;
	delete[] hiddenNegativeGradients;
}

Utility::Deltas::Deltas(double* w, double* v, double* h) :
	deltaWeightsMatrix(w), deltaVisibleBiases(v), deltaHiddenBiases(h) {}

Utility::Deltas::~Deltas() {
	delete[] deltaWeightsMatrix;
	delete[] deltaVisibleBiases;
	delete[] deltaHiddenBiases;
}
