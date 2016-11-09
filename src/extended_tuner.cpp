#include <chrono>
#include <iostream>
#include <memory>
#include <utility>

#include "extended_tuner.h"

namespace cltune
{

    ExtendedTuner::ExtendedTuner(size_t platformId, size_t deviceId, UniqueConfigurator configurator):
        basicTuner(new Tuner(platformId, deviceId))
    {
        this->configurator = std::move(configurator);
    } 

    ExtendedTuner::~ExtendedTuner() {}

    size_t ExtendedTuner::addKernel(const std::vector<std::string>& filenames, const std::string& kernelName, const IntRange& global,
                                    const IntRange& local)
    {
        return basicTuner->AddKernel(filenames, kernelName, global, local);
    }

    size_t ExtendedTuner::addKernelFromString(const std::string& source, const std::string& kernelName, const IntRange& global,
                                              const IntRange& local)
    {
        return basicTuner->AddKernelFromString(source, kernelName, global, local);
    }

    void ExtendedTuner::setReference(const std::vector<std::string>& filenames, const std::string& kernelName, const IntRange& global,
                                     const IntRange& local)
    {
        basicTuner->SetReference(filenames, kernelName, global, local);
    }

    void ExtendedTuner::setReferenceFromString(const std::string& source, const std::string& kernelName, const IntRange& global,
                                               const IntRange& local)
    {
        basicTuner->SetReferenceFromString(source, kernelName, global, local);
    }

    void ExtendedTuner::addParameter(const size_t id, const std::string& parameterName, const std::initializer_list<size_t>& values)
    {
        basicTuner->AddParameter(id, parameterName, values);
    }

    void ExtendedTuner::addParameterReference(const std::string& parameterName, const size_t value)
    {
        basicTuner->AddParameterReference(parameterName, value);
    }

    void ExtendedTuner::mulGlobalSize(const size_t id, const StringRange range)
    {
        basicTuner->MulGlobalSize(id, range);
    }

    void ExtendedTuner::divGlobalSize(const size_t id, const StringRange range)
    {
        basicTuner->DivGlobalSize(id, range);
    }

    void ExtendedTuner::mulLocalSize(const size_t id, const StringRange range)
    {
        basicTuner->MulLocalSize(id, range);
    }

    void ExtendedTuner::divLocalSize(const size_t id, const StringRange range)
    {
        basicTuner->DivLocalSize(id, range);
    }

    void ExtendedTuner::setMultirunKernelIterations(const size_t id, const std::string& parameterName)
    {
        basicTuner->SetMultirunKernelIterations(id, parameterName);
    }

    void ExtendedTuner::addConstraint(const size_t id, ConstraintFunction validIf, const std::vector<std::string>& parameters)
    {
        basicTuner->AddConstraint(id, validIf, parameters);
    }

    void ExtendedTuner::setLocalMemoryUsage(const size_t id, LocalMemoryFunction amount, const std::vector<std::string>& parameters)
    {
        basicTuner->SetLocalMemoryUsage(id, amount, parameters);
    }

    template <typename T> void ExtendedTuner::addArgumentInput(const size_t id, const std::vector<T> &source)
    {
        basicTuner->AddArgumentInput(id, source);
    }

    template <typename T> void ExtendedTuner::addArgumentOutput(const size_t id, const std::vector<T> &source)
    {
        basicTuner->AddArgumentOutput(id, source);
    }

    template <typename T> void ExtendedTuner::addArgumentScalar(const size_t id, const T argument)
    {
        basicTuner->AddArgumentScalar(id, argument);
    }

    template <typename T> void ExtendedTuner::addArgumentInputReference(const std::vector<T>& source)
    {
        basicTuner->AddArgumentInputReference(source);
    }

    template <typename T> void ExtendedTuner::addArgumentOutputReference(const std::vector<T>& source)
    {
        basicTuner->AddArgumentOutputReference(source);
    }
    template <typename T> void ExtendedTuner::addArgumentScalarReference(const T argument)
    {
        basicTuner->AddArgumentScalarReference(argument);
    }

    void ExtendedTuner::useFullSearch()
    {
        basicTuner->UseFullSearch();
    }

    void ExtendedTuner::useRandomSearch(const double fraction)
    {
        basicTuner->UseRandomSearch(fraction);
    }

    void ExtendedTuner::useAnnealing(const double fraction, const double maxTemperature)
    {
        basicTuner->UseAnnealing(fraction, maxTemperature);
    }

    void ExtendedTuner::usePSO(const double fraction, const size_t swarmSize, const double influenceGlobal, const double influenceLocal,
                               const double influenceRandom)
    {
        basicTuner->UsePSO(fraction, swarmSize, influenceGlobal, influenceLocal, influenceRandom);
    }

    void ExtendedTuner::chooseVerificationTechnique(const VerificationTechnique technique)
    {
        basicTuner->ChooseVerificationTechnique(technique);
    }

    void ExtendedTuner::chooseVerificationTechnique(const VerificationTechnique technique, const double toleranceTreshold)
    {
        basicTuner->ChooseVerificationTechnique(technique, toleranceTreshold);
    }

    void ExtendedTuner::outputSearchLog(const std::string &filename)
    {
        basicTuner->OutputSearchLog(filename);
    }

    void ExtendedTuner::modelPrediction(const Model modelType, const float validationFraction, const size_t testTopXConfigurations)
    {
        basicTuner->ModelPrediction(modelType, validationFraction, testTopXConfigurations);
    }

    double ExtendedTuner::printToScreen() const
    {
        return basicTuner->PrintToScreen();
    }

    void ExtendedTuner::printFormatted() const
    {
        basicTuner->PrintFormatted();
    }

    void ExtendedTuner::printJSON(const std::string& filename, const std::vector<std::pair<std::string, std::string>>& descriptions) const
    {
        basicTuner->PrintJSON(filename, descriptions);
    }

    void ExtendedTuner::printToFile(const std::string& filename) const
    {
        basicTuner->PrintToFile(filename);
    }

    void ExtendedTuner::runSingleKernel(const size_t id, const ParameterRange& parameterValues)
    {
        auto beforeTuningBegin = std::chrono::high_resolution_clock::now();
        configurator->beforeTuning();
        auto beforeTuningEnd = std::chrono::high_resolution_clock::now();
        auto beforeDuration = std::chrono::duration_cast<std::chrono::milliseconds>(beforeTuningEnd - beforeTuningBegin).count();

        float kernelDuration = basicTuner->RunSingleKernel(id, parameterValues);

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
