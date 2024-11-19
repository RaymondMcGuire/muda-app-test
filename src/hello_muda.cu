// To export hello_muda() function
// so that we could use it in main.cpp
#include "hello_muda.h"
#include <iostream>
#include <muda/muda.h>

// should not use in header file
#include <muda/syntax_sugar.h>

using namespace muda;

void hello_muda()
{
    std::cout << "Muda Runtime Check:" << muda::RUNTIME_CHECK_ON  << std::endl;

    DeviceVar<int> var = 1;

    Launch(1, 1)
        .kernel_name(__FUNCTION__)
        .apply(
        [
            var = var.viewer().name("var")
        ] __device__ () mutable
        {
            print("hello muda!\n"); 
            print("var = %d\n", var);
            print("var.kernel_name = %s\n", var.kernel_name());
            print("var.name = %s\n", var.name());
        }).wait();
}

void vec_add_basic(){

    std::cout << "Vec Add Basic Example"  << std::endl;

    constexpr int       N = 1024;
    HostVector<float>   hA(N), hB(N),hC(N);
    DeviceVector<float> dA(N), dB(N), dC(N);

    // initialize A and B using random numbers
    auto rand = [] { return std::rand() / (float)RAND_MAX; };
    std::generate(hA.begin(), hA.end(), rand);
    std::generate(hB.begin(), hB.end(), rand);

    // copy A and B to device
    dA = hA;
    dB = hB;

    int threadsPerBlock = 256;
    int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;

    Launch(blocksPerGrid, threadsPerBlock)
        .apply([dC = dC.viewer(),   // | this is a capture list              |
                dA = dA.cviewer(),  // | map from device_vector to a viewer  |
                dB = dB.cviewer()]  // | which is the most muda-style part!  |
               __device__() mutable  // place "mutable" to make dC modifiable
               {
                   int i = blockDim.x * blockIdx.x + threadIdx.x;
                   if(i < N)
                       dC(i) = dA(i) + dB(i);
               })
        .wait();  // wait the kernel to finish

    // copy C back to host
    hC = dC;

    std::cout << "Data Check:" << hA[0] << " + " << hB[0] << "=" << hC[0]<< std::endl;
}


void vec_add_parallel(){

    std::cout << "Vec Add Parallel Example"  << std::endl;

    constexpr int       N = 1024;
    HostVector<float>   hA(N), hB(N),hC(N);
    DeviceVector<float> dA(N), dB(N), dC(N);

    // initialize A and B using random numbers
    auto rand = [] { return std::rand() / (float)RAND_MAX; };
    std::generate(hA.begin(), hA.end(), rand);
    std::generate(hB.begin(), hB.end(), rand);

    // copy A and B to device
    dA = hA;
    dB = hB;

    int threadsPerBlock = 256;
    int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;

    // block size = 256
    ParallelFor(256)
        .kernel_name("vector_add")
        .apply(N,
               [$def(dC, dC.viewer()),   // | this is a capture list              |
                $def(dA, dA.ciewer()),  // | map from device_vector to a viewer  |
                $def(dB, dB.ciewer())]  // | which is the most muda-style part!  |
               __device__(int i) mutable  // place "mutable" to make dC modifiable
               {
                   // safe parallel for will cover the rang [0, N)
                   // i just goes from 0 to N-1
                   dC(i) = dA(i) + dB(i);
               })
        .wait();  // wait the kernel to finish

    // copy C back to host
    hC = dC;

    std::cout << "Data Check:" << hA[0] << " + " << hB[0] << "=" << hC[0]<< std::endl;
}