#include <iostream>
#include <ctime>
#include "RestrictedBoltzmannMachine.h"
#include "Utility.h"

int main()
{
	//TODO: read from console: samples, hidden=100, learningRate=0.05, cdSteps=10, miniBatch, epochs=20, mnist-path, save-path
	
	//Toy example
	//RestrictedBoltzmannMachine rbm = RestrictedBoltzmannMachine(3, 2);
	//double data[10] = { 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0 };
	//
	//std::cout << rbm.GetWeights() << std::endl;
	//rbm.TrainNetwork(data, 5, 0.1, 5, 1, 1);
	//std::cout << rbm.GetWeights() << std::endl;
	//rbm.SaveWeights("learned_weights.txt");

	// Real example
	// Read data to train
	const auto samples = 10000;
	const int visible = 784;
	double* data = Utility::readMnistImagesBinarized("D:\\FinalRBM\\SolutionRBM\\x64\\Release\\train-images-idx3-ubyte", samples);

	// Train model (full batch)	
	const int hidden = 100;
	auto start = clock();
	RestrictedBoltzmannMachine rbm = RestrictedBoltzmannMachine(hidden, visible);
	rbm.TrainNetwork(data, samples, 0.05, samples, 20, 10);
	auto end = clock();

	// Results and cleaning
	rbm.SaveWeights("learned_weights_gpu.txt");
	std::cout << "Time elapsed: " << (end - start) / CLOCKS_PER_SEC << "s" << std::endl;
	delete[] data;


	system("PAUSE");
    return 0;
}