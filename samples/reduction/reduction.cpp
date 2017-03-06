//#include <libc.h>
#include <stdbool.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <CL/opencl.h>
//#include <mach/mach_time.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <string>
#include <vector>

#include "extended_tuner.h"

/////////////////////////////////////////////////////////////////////////////
const char* TUNED_KERNEL_NAME = "reduction.cl";
const char* REFERENCE_KERNEL_NAME = "reduction_reference.cl";

static int size       = 1024*1024*128;

cl_device_id getDeviceID(int platformIndex, int deviceIndex) {
    cl_int ciErr;
    cl_platform_id cpPlatform[4];
    cl_uint platforms = 0;
    ciErr = clGetPlatformIDs(4, cpPlatform, &platforms);
    if (ciErr != CL_SUCCESS)
    {
        fprintf(stderr, "Error in clGetPlatformID.");
        return NULL;
    }
    if (platformIndex >= platforms) {
        fprintf(stderr, "Error: requested platform does not exist.");
        return NULL;
    }
    cl_uint devices = 0;
    cl_device_id cdDevice[4];
    ciErr = clGetDeviceIDs(cpPlatform[platformIndex], CL_DEVICE_TYPE_ALL, 4, cdDevice, &devices);
    if (ciErr != CL_SUCCESS)
    {
        fprintf(stderr, "Error in clGetDeviceIDs.");
        return NULL;
    }
    if (deviceIndex >= devices) {
        fprintf(stderr, "Error: requested device does not exist.");
        return NULL;
    }

    return cdDevice[deviceIndex];
}

void printDeviceInfo(cl_device_id device_id) {
    char device_string[1024];
    clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_string), &device_string, NULL);
    printf(" CL_DEVICE_NAME: %s\n", device_string);
    clGetDeviceInfo(device_id, CL_DEVICE_VENDOR, sizeof(device_string), &device_string, NULL);
    printf(" CL_DEVICE_VENDOR: %s\n", device_string);
}

size_t getComputeUnitNum(cl_device_id device_id) {
    int ret;
    clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(ret), &ret, NULL);
    return size_t(ret);
}

/////////////////////////////////////////////////////////////////////////////

class ReductionConfigurator : public cltune::TunerConfigurator
{
public:
    ReductionConfigurator(cltune::ExtendedTuner& tuner, size_t kernelId) :
        kernelId(kernelId)
    {
        this->tuner = &tuner;
    }

    virtual cltune::PublicTunerResult customizedComputation(const cltune::ParameterRange& configuration)
    {
        //tuner->
        return tuner->runSingleKernel(kernelId, configuration);
    }

private:
    cltune::ExtendedTuner* tuner;
    size_t kernelId;
};

int main(int argc, char **argv)
{
    std::vector<float> dst(size); // large enough :-)
    //std::vector<float> dst(1);
    std::vector<float> src(size);
    std::vector<size_t> ndRangeDimensions{ (size_t)size };

    srand(0);
    // fill vector with random values from <0,1>
    for (int i = 0; i < size; i++)
    {
        src[i] = (float) rand() / (float) RAND_MAX;
        dst[i] = 0.0f;
    }
    //dst[0] = 0.0f;

    int platformIndex = 0;
    int deviceIndex = 0;
    if (argc >= 2)
    {
        platformIndex = std::stoi(std::string{ argv[1] });
        if (argc >= 3)
        {
            deviceIndex = std::stoi(std::string{ argv[2] });
        }
    }

/*    cl_platform_id cpPlatform[4];
    cl_uint platforms = 0;
    ciErr1 = clGetPlatformIDs(4, cpPlatform, &platforms);
    if (ciErr1 != CL_SUCCESS)
    {
        fprintf(stderr, "Error in clGetPlatformID.");
        Cleanup(argc, argv, EXIT_FAILURE);
    }
    cl_device_id cdDevice[4];
    ciErr1 = clGetDeviceIDs(cpPlatform[platformIndex], CL_DEVICE_TYPE_ANY, 1, &cdDevice, 4);
*/
    cl_device_id deviceId = getDeviceID(platformIndex, deviceIndex);    
    printDeviceInfo(deviceId);
    size_t cus = (size_t)getComputeUnitNum(deviceId);
    printf("Number of CUs: %i\n", (int)cus);

    cltune::Tuner tuner(platformIndex, deviceIndex);
    size_t kernelId = tuner.AddKernel(std::vector<std::string>{ TUNED_KERNEL_NAME }, "reduce", ndRangeDimensions, { 1 });
    tuner.AddParameter(kernelId, "WORK_GROUP_SIZE_X", { /*1, 2, 4, 8, 16, 32,*/ 64, 128, 256, 512 });
    tuner.AddParameter(kernelId, "UNBOUNDED_WG", {0, 1});
    tuner.AddParameter(kernelId, "WG_NUM", {0, cus, cus*2, cus*4, cus*8, cus*16});
    tuner.AddParameter(kernelId, "VECTOR_SIZE", {1, 2, 4, 8, 16});
    tuner.AddParameter(kernelId, "USE_ATOMICS", {0, 1});
    // set local size to WORK_GROUP_SIZE_X
    tuner.MulLocalSize(kernelId, { "WORK_GROUP_SIZE_X" });
    // set global size according to WG persistency
    tuner.DivGlobalSize(kernelId, { "VECTOR_SIZE" }); // divide by vector size
    tuner.DivGlobalSize(kernelId, { "WORK_GROUP_SIZE_X" }); // convert size to WG num
    tuner.MulGlobalSize(kernelId, { "UNBOUNDED_WG" }); // sets to 0 for persisten WG, not modify otherwise
    //tuner.AddGlobalSize(kernelId, { "WG_NUM" }); // add number of persistent WGs (0 if persistency not used)
    tuner.MulGlobalSize(kernelId, { "WORK_GROUP_SIZE_X" }); // return from WG num to global size
    auto persistConstraint = [] (std::vector<size_t> v) { return (v[0] && v[1] == 0) || (!v[0] && v[1] > 0); };
    tuner.AddConstraint(kernelId, persistConstraint, { "UNBOUNDED_WG", "WG_NUM" });

    tuner.SetReference(std::vector<std::string>{ REFERENCE_KERNEL_NAME }, "reduceReference", ndRangeDimensions, { 256 });

    tuner.AddArgumentInput(kernelId, src);
    tuner.AddArgumentOutput(kernelId, dst);
    tuner.AddArgumentScalar(kernelId, size);

    tuner.TuneAllKernels();
    tuner.PrintToScreen();
    tuner.PrintToFile("result.csv");

    return 0;
}
