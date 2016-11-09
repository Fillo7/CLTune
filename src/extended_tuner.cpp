#include <chrono>
#include <iostream>
#include <memory>
#include <utility>

#include "extended_tuner.h"

namespace cltune
{

    ExtendedTuner::ExtendedTuner(UniqueTuner basicTuner, UniqueConfigurator configurator)
    {
        this->basicTuner = std::move(basicTuner);
        this->configurator = std::move(configurator);
    } 

    ExtendedTuner::~ExtendedTuner() {}

    void ExtendedTuner::runSingleKernel(const size_t id, const ParameterRange &parameter_values)
    {
        auto beforeTuningBegin = std::chrono::high_resolution_clock::now();
        configurator->beforeTuning();
        auto beforeTuningEnd = std::chrono::high_resolution_clock::now();
        auto beforeDuration = std::chrono::duration_cast<std::chrono::milliseconds>(beforeTuningEnd - beforeTuningBegin).count();

        float kernelDuration = basicTuner->RunSingleKernel(id, parameter_values);

        auto afterTuningBegin = std::chrono::high_resolution_clock::now();
        configurator->afterTuning();
        auto afterTuningEnd = std::chrono::high_resolution_clock::now();
        auto afterDuration = std::chrono::duration_cast<std::chrono::milliseconds>(afterTuningEnd - afterTuningBegin).count();

        std::cout << "[Extended Tuner]" << "Duration of beforeTuning() method: " << beforeDuration << "ms." << std::endl;
        std::cout << "[Extended Tuner]" << "Duration of kernel execution: " << kernelDuration << "ms." << std::endl;
        std::cout << "[Extended Tuner]" << "Duration of afterTuning() method: " << afterDuration << "ms." << std::endl;
        std::cout << "[Extended Tuner]" << "Total duration: " << beforeDuration + afterDuration + kernelDuration << "ms." << std::endl;
    }

    void ExtendedTuner::tune()
    {
        auto beforeTuningBegin = std::chrono::high_resolution_clock::now();
        configurator->beforeTuning();
        auto beforeTuningEnd = std::chrono::high_resolution_clock::now();
        auto beforeDuration = std::chrono::duration_cast<std::chrono::milliseconds>(beforeTuningEnd - beforeTuningBegin).count();

        float bestKernelDuration = basicTuner->Tune();

        auto afterTuningBegin = std::chrono::high_resolution_clock::now();
        configurator->afterTuning();
        auto afterTuningEnd = std::chrono::high_resolution_clock::now();
        auto afterDuration = std::chrono::duration_cast<std::chrono::milliseconds>(afterTuningEnd - afterTuningBegin).count();

        std::cout << "[Extended Tuner]" << "Duration of beforeTuning() method: " << beforeDuration << "ms." << std::endl;
        std::cout << "[Extended Tuner]" << "Duration of fastest kernel execution: " << bestKernelDuration << "ms." << std::endl;
        std::cout << "[Extended Tuner]" << "Duration of afterTuning() method: " << afterDuration << "ms." << std::endl;
        std::cout << "[Extended Tuner]" << "Total duration: " << beforeDuration + afterDuration + bestKernelDuration << "ms." << std::endl;
    }

} // namespace cltune
