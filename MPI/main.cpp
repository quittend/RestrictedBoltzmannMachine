#include <iostream>
#include <ctime>
#include <math.h>
#include <iostream>
#include "RestrictedBoltzmannMachine.h"
#include "Utility.h"
#include <mpi.h>
#include <fstream>

int main() {
    const auto samples = 10000;
    const int visible = 784;
    const int hidden = 100;

    MPI_Init(NULL, NULL);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int workers_count = world_size -1;
    int batch_size = ceil(double(samples) / double(workers_count));

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if(world_rank == 0) {  // master process
        std::shared_ptr<double> data = Utility::readMnistImagesBinarized("/home/vito/workspace/PORR/RestrictedBoltzmannMachineMPI/mnist/train-images-idx3-ubyte", samples);

        auto start = clock();
        // divide set, send batches to workers
        unsigned int offset = 0;
        unsigned int size = batch_size;

        for(auto i=0; i<workers_count; ++i) {
            if((i == samples%workers_count) && (i != 0)) {
                size = batch_size - 1;
            }
            LOG("M  sending batch " << size << " to worker " << i+1 << "\n");
            MPI_Send(data.get() + offset*visible, size*visible, MPI_DOUBLE, i+1, Utility::SND_BATCH, MPI_COMM_WORLD);
            offset += size;
        }

        // receive results from worker 1
        std::shared_ptr<double> buffer = std::shared_ptr<double>(new double[hidden*visible], std::default_delete<double[]>());
        LOG("M  receiving result\n");
        MPI_Recv(buffer.get(), hidden*visible, MPI_DOUBLE, 1, Utility::SND_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        auto end = clock();
        Utility::saveWeights("learned_weights_mpi.txt", buffer, visible, hidden);
        std::cout << "Time elapsed: " << (end - start) / CLOCKS_PER_SEC << "s" << std::endl;

    } else {  // worker process
        // receive batch
        std::shared_ptr<double> buffer = std::shared_ptr<double>(new double[batch_size*visible], std::default_delete<double[]>());
        LOG("W" << world_rank << " receiving batch\n");
        MPI_Recv(buffer.get(), batch_size*visible, MPI_DOUBLE, 0, Utility::SND_BATCH, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        int my_batch_size;
        if((world_rank -1 >= samples%workers_count) && (samples%workers_count != 0)) {
            my_batch_size = batch_size - 1;
        } else {
            my_batch_size = batch_size;
        }

        // train batch
        LOG("W" << world_rank << " training batch\n");
        RestrictedBoltzmannMachine rbm = RestrictedBoltzmannMachine(hidden, visible);

        rbm.TrainBatch(buffer, my_batch_size, 0.05, my_batch_size, 20, 10);
        LOG("W" << world_rank << " trained\n");
        if(world_rank == 1) {
            // send results to master
            LOG("W" << world_rank << " sending result\n");
            MPI_Send(rbm.getWeights().get(), hidden*visible, MPI_DOUBLE, 0, Utility::SND_RESULT, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();

    return 0;
}