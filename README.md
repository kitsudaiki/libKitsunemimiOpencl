# libKitsunemimiOpencl

## IMPORTANT: This repository is no longer maintained, because in context of issue https://github.com/kitsudaiki/Hanami-AI/issues/31 the content was moved into the main-repository (https://github.com/kitsudaiki/Hanami-AI) in order to reduce the maintenance workload.

![Github workfloat status](https://img.shields.io/github/actions/workflow/status/kitsudaiki/libKitsunemimiOpencl/build_test.yml?branch=develop&style=flat-square&label=build%20and%20test)
![GitHub tag (latest SemVer)](https://img.shields.io/github/v/tag/kitsudaiki/libKitsunemimiOpencl?label=version&style=flat-square)
![GitHub](https://img.shields.io/github/license/kitsudaiki/libKitsunemimiOpencl?style=flat-square)
![C++Version](https://img.shields.io/badge/c%2B%2B-17-blue?style=flat-square)
![Platform](https://img.shields.io/badge/platform-Linux--x64-lightgrey?style=flat-square)

## Description

This is a simple wrapper-library for opencl. It provides some basic commenly used functionalities and abstract them to a very simple structure, which comes with some restrictions, but make it fast and easy to write a program with opencl support. The basic feature set provides the following functions:

- init device
- copy data to device
- run kernel
- copy data from device
- close device
- get work-item information from device
- get memory information from device

These abstractions have some restrictions like for example the follwing:

- all copy-transfers are always blocking
- always the whole buffer is copied to or from device. copy only parts of the buffer is not supported
- only one device handable at the moment or more precisely the first one, which is found on the system.

To avoid restrictions for special cases, all opencl-objects are public and so normal opencl-operations can also performed without the abstracted functions of this library.

## Build

### Requirements

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | 8.x | Compiler for the C++ code.
qmake | qt5-qmake | 5.x | This package provides the tool qmake, to build the project
opencl-headers  | opencl-headers | 2.x | Header-files for opencl
ocl-icd-opencl-dev | ocl-icd-opencl-dev | 2.x | libraries for opencl

Installation on Ubuntu/Debian:

```bash
sudo apt-get install g++ qt5-qmake opencl-headers ocl-icd-opencl-dev
```

Based on the used gpu, there is an additional package necessary:

- Intel: `intel-opencl-icd` and requirements of these packages
- AMD: `opencl-amdgpu-pro-icd`, `opencl-amdgpu-pro`, `libopencl1-amdgpu-pro` and requirements of these packages
- Nvidia: `nvidia-opencl-icd`, `nvidia-libopencl1` and requirements of these packages (not tested from my side)


IMPORTANT: All my projects are only tested on Linux. 

### Kitsunemimi-repositories

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.23.0 |  https://github.com/kitsudaiki/libKitsunemimiCommon.git

HINT: These Kitsunemimi-Libraries will be downloaded and build automatically with the build-script below.

### build library

In all of my repositories you will find a `build.sh`. You only have to run this script. It doesn't required sudo, because you have to install required tool via apt, for example, by yourself. But if other projects from me are required, it download them from github and build them in the correct version too. This script is also use by the ci-pipeline, so its tested with every commit.


Run the following commands:

```
git clone https://github.com/kitsudaiki/libKitsunemimiOpencl.git
cd libKitsunemimiOpencl
./build.sh
cd ../result
```

It create automatic a `build` and `result` directory in the directory, where you have cloned the project. At first it build all into the `build`-directory and after all build-steps are finished, it copy the include directory from the cloned repository and the build library into the `result`-directory. So you have all in one single place.

Tested on Debian and Ubuntu. If you use Centos, Arch, etc and the build-script fails on your machine, then please write me a mail and I will try to fix the script.


## Usage

Here only an example kernel code to 

```cpp
const std::string kernelCode =
    "__kernel void test_kernel(\n"
    "       __global const float* a,\n"  // <-- first input-buffer a
    "       __global float* b\n"        // <-- output-buffer
    "       )\n"
    "{\n"
    "    // do something with the data on device. This here is only a stupid useless example."
    "    size_t globalId = get_global_id(0) + get_global_size(0) * get_global_id(1);\n"
    "    if (globalId < N)\n"
    "    {\n"
    "       b[globalId] = a[globalId];"
    "    }\n"
    "}\n";
```

The copy process for the library from host to device copies always at first the buffer and then the number of elements of this buffer. So they are always pairwise in the arguments of the kernel-function.


```cpp
#include <libKitsunemimiOpencl/gpu_handler.h>
#include <libKitsunemimiOpencl/gpu_interface.h>
#include <libKitsunemimiCommon/logger.h>

// Optional  initialize the logger. This here initalize a console logger, 
// which prints all error- and info-messages on the consol
Kitsunemimi::initConsoleLogger(true);
Kitsunemimi::ErrorContainer error;
// in case of an error the message con be printed with LOG_ERROR(error)

// init opencl-class of this library
Kitsunemimi::GpuHandler oclHandler;
oclHandler.initDevice(error)
// the GpuHandler collect all devices of the host and stores them 
// into oclHandler.m_interfaces

// get for example the first device
Kitsunemimi::GpuInterface* ocl = oclHandler.m_interfaces.at(0);
// this ocl-object here will be used in all the following snippets
```

Prepare buffer for data-transfers between the host and the device.

```cpp
// create data-object
Kitsunemimi::GpuData data;

// init empty buffer
// This prepare a buffer-object and allocate aligned memorey on the host.
// These objects will be used to transfer data between the host and the device.
data.addBuffer("buffer x",     // <-- self-defined id for the buffer
               N,              // <-- number of elements
               sizeof(float),  // <-- size of one element
               true);          // <-- set to true use a host-pointer
                               //     This makes copy to device faster,
                               //     but the kernel will be slower.
                               //     So keep the tradeoff in mind!
data.addBuffer("buffer y",
               N, 
               sizeof(float), 
               false);
// in the same style, there are multiple input- and output-buffer possible

// for the example get here the first buffer and set all values of this buffer to 1.0
float* a = static_cast<float*>(data.getBufferData("buffer x"));
for(uint32_t i = 0; i < N; i++) {
    a[i] = 1.0f;
}
```

Init worker-sizes. Tehre are two fields: number of work-groups and threads per work-group. Normally this are global and local work-items in opencl, but I wanted it a little bit more like in CUDA.

```cpp
data.numberOfWg.x = N / 512;
data.numberOfWg.y = 2;
data.threadsPerWg.x = 256;
```

In the main-part copy the data to the device and process them on the device.

```cpp
bool ret = false;

// copy the data of OpenClData-object, which was initialized in the snipped before 
ret = ocl->initCopyToDevice(data, error);

// add kernel-code with name to device
ret = ocl->addKernel(data, "test_kernel", kernelCode, error)
// you can all multiple kernel to the device and its queue

// bind buffer 0 and 1 to the kernel
ret = ocl->bindKernelToBuffer(data, "test_kernel", "buffer x", error);
ret = ocl->bindKernelToBuffer(data, "test_kernel", "buffer y", error);

// updata this on the host changed buffer also on the device with the following command
ocl->updateBufferOnDevice(data, "buffer x", error);

// run kernel-code this the data
ret = ocl->run(data, "test_kernel", error);

// copy all as output-buffer defined buffer from device back to host
ret = ocl->copyFromDevice(data, "buffer y", error);

// access the data in the output-buffer and process the result on the host
float* outputValues = static_cast<float*>(data.getBufferData("buffer y"));
```

Maybe you want to make more then one run. So you can update all buffer on the device, which are NOT defined as output-buffer.

```cpp
bool ret = false;

// for this example get the first buffer again and change this content
float* a = static_cast<float*>(data.getBufferData("buffer x"));
for(uint32_t i = 0; i < N; i++) {
    a[i] = 2.0f;
}

// updata this on the host changed buffer also on the device with the following command
ocl->updateBufferOnDevice(data, "buffer x", error);

// run the kernel again and copy the result back again.
ret = ocl->run(data);
ret = ocl->copyFromDevice(data);

float* outputValues = static_cast<float*>(data.getBufferData("buffer y"));
```

It is also possible to get some basic information from these opencl-wrapper-class. These getter are restricted for the available memory on the device and the maximum sizes of the worker-groups. 

```cpp
// getter for memory
uint64_t localSize = ocl->getLocalMemorySize();
uint64_t globalSize = ocl->getGlobalMemorySize();
uint64_t maxAlloc = ocl->getMaxMemAllocSize();

// getter for work-groups sizes
uint64_t maxWorkGroupSize = ocl->getMaxWorkGroupSize();
uint64_t maxWorkItemDimension = ocl->getMaxWorkItemDimension();
WorkerDim dim = ocl->getMaxWorkItemSize();
```

If you need other information, which are not covered by this few getter here, you can perform normal operations on the device-objects. For example like this:

```cpp
cl_ulong size = 0;
ocl->m_device.at(0).getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &size);
```

After all was done, then close the device.

```cpp
// close the device and free all buffer. For this it requires the data-object
// with the input- and output-data to free the buffer on the device and free the
// allocated memory inside these data-objects
ret = ocl->closeDevice(data);

// the destructor of the Opencl-class, where this ocl-object belongs to, also calls
// this method, but without the data-object, so the destructor doesn't free the memory.
```

## Contributing

Please give me as many inputs as possible: Bugs, bad code style, bad documentation and so on.

## License

This project is licensed under the Apache License Version 2.0 - see the [LICENSE](LICENSE) file for details

