#include <cstdlib>
#include <ctime>
#include <iostream>
#include <memory>
#include <vector>

#include "extended_tuner.h"

// Inline definition of example configurator class, which provides implementation of TunerConfigurator interface
/*class ExampleConfigurator : public cltune::TunerConfigurator
{
public:
ExampleConfigurator() {}

ExampleConfigurator(cltune::ExtendedTuner& tuner, size_t kernelId):
kernelId(kernelId)
{
this->tuner = &tuner;
}

virtual void beforeTuning()
{
customBeforeTuning(4);
}

virtual void afterTuning()
{
std::cout << "afterTuning() method of example configurator is in progress..." << std::endl;
}

private:
cltune::ExtendedTuner* tuner;
size_t kernelId;

void customBeforeTuning(int customAttribute)
{
tuner->addArgumentScalar(kernelId, customAttribute);
std::cout << "Custom beforeTuning() method of example configurator is in progress. Custom attribute is being used: " << customAttribute << std::endl;
}
};*/

int main(int argc, char** argv)
{
    // Initialize platform and device index
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

    // Declare constants
    const float UPPER_INTERVAL_BOUNDARY = 1000.0f; // used for generating random test data
    const std::string multiRunKernelName = std::string("multirunKernel.cl");
    const std::string referenceKernelName = std::string("referenceKernel.cl");

    // Declare kernel parameters
    const int numberOfElements = 4096 * 4096;
    std::vector<size_t> ndRangeDimensions{ 4096 * 4096, 1 };
    std::vector<size_t> workGroupDimensions{ 256, 1 };

    // Declare data variables
    std::vector<float> a(numberOfElements);
    std::vector<float> b(numberOfElements);
    std::vector<float> result(numberOfElements, 0.0f);

    // Initialize data
    srand((unsigned)time(0));

    for (int i = 0; i < numberOfElements; i++)
    {
        a.at(i) = (float)rand() / (RAND_MAX / UPPER_INTERVAL_BOUNDARY);
        b.at(i) = (float)rand() / (RAND_MAX / UPPER_INTERVAL_BOUNDARY);
    }

    // Kernel tuning section
    cltune::ExtendedTuner tuner(platformIndex, deviceIndex);
    size_t kernelId = tuner.addKernel(std::vector<std::string> { multiRunKernelName }, "multirunKernel", ndRangeDimensions, workGroupDimensions);

    // Add parameter, which will specify the number of iterations the kernel will run through, multiple values are possible for purpose of autotuning
    tuner.addParameter(kernelId, "VALID_MULTIRUNS", { 1, 2, 4, 8 });

    // This method simply ties the previously added parameter to the number of kernel iterations
    tuner.setMultirunKernelIterations(kernelId, "VALID_MULTIRUNS");

    // It is necessary to divide NDRange size by the number of iterations, because with higher number of iterations, the NDRange size is lower
    // in each separate kernel run
    tuner.addParameter(kernelId, "ALWAYS_ONE", { 1 }); // just a dummy parameter, second dimension of NDRange doesn't need to be divided
    tuner.divGlobalSize(kernelId, { "VALID_MULTIRUNS", "ALWAYS_ONE" });

    // Reference kernel has to always run in single iteration
    tuner.setReference(std::vector<std::string>{ referenceKernelName }, "referenceKernel", ndRangeDimensions, workGroupDimensions);

    // Each kernel takes separate arguments
    tuner.addArgumentScalar(kernelId, 2.0f);
    tuner.addArgumentInput(kernelId, a);
    tuner.addArgumentInput(kernelId, b);
    tuner.addArgumentOutput(kernelId, result);

    // Different method is used for adding arguments to reference kernel, output buffers for all kernels should have the same size so the results
    // comparison is possible
    tuner.addArgumentScalarReference(2.0f);
    tuner.addArgumentInputReference(a);
    tuner.addArgumentInputReference(b);
    tuner.addArgumentOutputReference(result);

    // Explicitly choose full search option (this is here to show correct usage, full search is default option so calling this is not strictly necessary)
    tuner.useFullSearch(kernelId);

    // Choose verification technique and specify tolerance treshold
    tuner.chooseVerificationTechnique(cltune::VerificationTechnique::SideBySide, 1e-4);

    // Set the tuner configurator to newly created custom class (currently unusable)
    //tuner.setConfigurator(kernelId, cltune::UniqueConfigurator(new ExampleConfigurator(tuner, kernelId)));

    // Begin tuning process for all kernels
    tuner.tuneAllKernels();

    // Print tuning results for all kernels to screen and to file
    tuner.printToScreen();
    tuner.printToFile(std::string("test.txt"));

    std::cin.get();
    return 0;
}
