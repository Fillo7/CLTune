#include <chrono>
#include <iostream>
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
        auto beforeDuration = std::chrono::duration_cast<std::chrono::nanoseconds>(beforeTuningEnd - beforeTuningBegin).count();

        basicTuner->RunSingleKernel(id, parameter_values);

        auto afterTuningBegin = std::chrono::high_resolution_clock::now();
        configurator->afterTuning();
        auto afterTuningEnd = std::chrono::high_resolution_clock::now();
        auto afterDuration = std::chrono::duration_cast<std::chrono::nanoseconds>(afterTuningEnd - afterTuningBegin).count();

        std::cout << "Duration of beforeTuning() method: " << beforeDuration << "ns." << std::endl;
        std::cout << "Duration of kernel execution: " << "<to do>" << "ns." << std::endl;
        std::cout << "Duration of afterTuning() method: " << afterDuration << "ns." << std::endl;
        std::cout << "Total duration: " << beforeDuration + afterDuration /* + kernel execution time */ << "ns." << std::endl;
    }

    void ExtendedTuner::tune()
    {
        auto beforeTuningBegin = std::chrono::high_resolution_clock::now();
        configurator->beforeTuning();
        auto beforeTuningEnd = std::chrono::high_resolution_clock::now();
        auto beforeDuration = std::chrono::duration_cast<std::chrono::nanoseconds>(beforeTuningEnd - beforeTuningBegin).count();

        basicTuner->Tune();

        auto afterTuningBegin = std::chrono::high_resolution_clock::now();
        configurator->afterTuning();
        auto afterTuningEnd = std::chrono::high_resolution_clock::now();
        auto afterDuration = std::chrono::duration_cast<std::chrono::nanoseconds>(afterTuningEnd - afterTuningBegin).count();

        std::cout << "Duration of beforeTuning() method: " << beforeDuration << "ns." << std::endl;
        std::cout << "Duration of fastest kernel execution: " << "<to do>" << "ns." << std::endl;
        std::cout << "Duration of afterTuning() method: " << afterDuration << "ns." << std::endl;
        std::cout << "Total duration: " << beforeDuration + afterDuration /* + fastest kernel execution time */ << "ns." << std::endl;
    }
} // namespace cltune
