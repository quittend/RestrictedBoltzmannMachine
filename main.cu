#include <iostream>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include "RestrictedBoltzmannMachine.cuh"
#include "Utility.cuh"
#include <ctime>

int main() {
	//TODO: read from console: samples, hidden=100, learningRate=0.05, cdSteps=10, miniBatch, epochs=20, mnist-path, save-path

	//Toy example
	//RestrictedBoltzmannMachine rbm = RestrictedBoltzmannMachine(3, 2);
	//double data[10] = { 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0 };
	//double* dataDevice; cudaMalloc(&dataDevice, sizeof(double) * 10);
	//cudaMemcpy(dataDevice, data, sizeof(double) * 10, cudaMemcpyHostToDevice);

	//std::cout << rbm.GetWeights() << std::endl;
	//rbm.TrainNetwork(dataDevice, 5, 0.05, 5, 1, 1);
	//std::cout << rbm.GetWeights() << std::endl;


	// Real example
	// Read data to train
	const auto samples = 10;
	const int visible = 784;
	double* data = Utility::readMnistImagesBinarized("D:\\FinalRBM\\SolutionRBM\\x64\\Release\\train-images-idx3-ubyte", samples);
	double* dataDevice; cudaMalloc(&dataDevice, sizeof(double)*samples*visible);
	cudaMemcpy(dataDevice, data, sizeof(double)*samples*visible, cudaMemcpyHostToDevice);

	// Train model (full batch)
	const int hidden = 100;
	auto start = clock();
	RestrictedBoltzmannMachine rbm = RestrictedBoltzmannMachine(hidden, visible);
	rbm.TrainNetwork(dataDevice, samples, 0.05, samples, 20, 10);
	auto end = clock();

	// Results and cleaning
	rbm.SaveWeights("learned_weights_gpu.txt");
	std::cout << "Time elapsed: " << (end - start) / CLOCKS_PER_SEC << "s" << std::endl;
	delete[] data;

	system("PAUSE");
	return 0;
}