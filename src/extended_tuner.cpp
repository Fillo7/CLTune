#include <chrono>
#include <complex>
#include <iostream>
#include <memory>
#include <utility>

#include "CL/cl.h"
#include "extended_tuner.h"

namespace cltune
{
    using half = cl_half;
    using float2 = std::complex<float>;
    using double2 = std::complex<double>;

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

    template void PUBLIC_API ExtendedTuner::addArgumentInput<short>(const size_t id, const std::vector<short>&);
    template void PUBLIC_API ExtendedTuner::addArgumentInput<int>(const size_t id, const std::vector<int>&);
    template void PUBLIC_API ExtendedTuner::addArgumentInput<size_t>(const size_t id, const std::vector<size_t>&);
    template void PUBLIC_API ExtendedTuner::addArgumentInput<half>(const size_t id, const std::vector<half>&);
    template void PUBLIC_API ExtendedTuner::addArgumentInput<float>(const size_t id, const std::vector<float>&);
    template void PUBLIC_API ExtendedTuner::addArgumentInput<double>(const size_t id, const std::vector<double>&);
    template void PUBLIC_API ExtendedTuner::addArgumentInput<float2>(const size_t id, const std::vector<float2>&);
    template void PUBLIC_API ExtendedTuner::addArgumentInput<double2>(const size_t id, const std::vector<double2>&);

    template <typename T> void ExtendedTuner::addArgumentOutput(const size_t id, const std::vector<T> &source)
    {
        basicTuner->AddArgumentOutput(id, source);
    }

    template void PUBLIC_API ExtendedTuner::addArgumentOutput<short>(const size_t id, const std::vector<short>&);
    template void PUBLIC_API ExtendedTuner::addArgumentOutput<int>(const size_t id, const std::vector<int>&);
    template void PUBLIC_API ExtendedTuner::addArgumentOutput<size_t>(const size_t id, const std::vector<size_t>&);
    template void PUBLIC_API ExtendedTuner::addArgumentOutput<half>(const size_t id, const std::vector<half>&);
    template void PUBLIC_API ExtendedTuner::addArgumentOutput<float>(const size_t id, const std::vector<float>&);
    template void PUBLIC_API ExtendedTuner::addArgumentOutput<double>(const size_t id, const std::vector<double>&);
    template void PUBLIC_API ExtendedTuner::addArgumentOutput<float2>(const size_t id, const std::vector<float2>&);
    template void PUBLIC_API ExtendedTuner::addArgumentOutput<double2>(const size_t id, const std::vector<double2>&);

    template <typename T> void ExtendedTuner::addArgumentScalar(const size_t id, const T argument)
    {
        basicTuner->AddArgumentScalar(id, argument);
    }

    template void PUBLIC_API ExtendedTuner::addArgumentScalar<short>(const size_t id, const short argument);
    template void PUBLIC_API ExtendedTuner::addArgumentScalar<int>(const size_t id, const int argument);
    template void PUBLIC_API ExtendedTuner::addArgumentScalar<size_t>(const size_t id, const size_t argument);
    template void PUBLIC_API ExtendedTuner::addArgumentScalar<half>(const size_t id, const half argument);
    template void PUBLIC_API ExtendedTuner::addArgumentScalar<float>(const size_t id, const float argument);
    template void PUBLIC_API ExtendedTuner::addArgumentScalar<double>(const size_t id, const double argument);
    template void PUBLIC_API ExtendedTuner::addArgumentScalar<float2>(const size_t id, const float2 argument);
    template void PUBLIC_API ExtendedTuner::addArgumentScalar<double2>(const size_t id, const double2 argument);

    template <typename T> void ExtendedTuner::addArgumentInputReference(const std::vector<T>& source)
    {
        basicTuner->AddArgumentInputReference(source);
    }

    template void PUBLIC_API ExtendedTuner::addArgumentInputReference<short>(const std::vector<short>&);
    template void PUBLIC_API ExtendedTuner::addArgumentInputReference<int>(const std::vector<int>&);
    template void PUBLIC_API ExtendedTuner::addArgumentInputReference<size_t>(const std::vector<size_t>&);
    template void PUBLIC_API ExtendedTuner::addArgumentInputReference<half>(const std::vector<half>&);
    template void PUBLIC_API ExtendedTuner::addArgumentInputReference<float>(const std::vector<float>&);
    template void PUBLIC_API ExtendedTuner::addArgumentInputReference<double>(const std::vector<double>&);
    template void PUBLIC_API ExtendedTuner::addArgumentInputReference<float2>(const std::vector<float2>&);
    template void PUBLIC_API ExtendedTuner::addArgumentInputReference<double2>(const std::vector<double2>&);

    template <typename T> void ExtendedTuner::addArgumentOutputReference(const std::vector<T>& source)
    {
        basicTuner->AddArgumentOutputReference(source);
    }

    template void PUBLIC_API ExtendedTuner::addArgumentOutputReference<short>(const std::vector<short>&);
    template void PUBLIC_API ExtendedTuner::addArgumentOutputReference<int>(const std::vector<int>&);
    template void PUBLIC_API ExtendedTuner::addArgumentOutputReference<size_t>(const std::vector<size_t>&);
    template void PUBLIC_API ExtendedTuner::addArgumentOutputReference<half>(const std::vector<half>&);
    template void PUBLIC_API ExtendedTuner::addArgumentOutputReference<float>(const std::vector<float>&);
    template void PUBLIC_API ExtendedTuner::addArgumentOutputReference<double>(const std::vector<double>&);
    template void PUBLIC_API ExtendedTuner::addArgumentOutputReference<float2>(const std::vector<float2>&);
    template void PUBLIC_API ExtendedTuner::addArgumentOutputReference<double2>(const std::vector<double2>&);

    template <typename T> void ExtendedTuner::addArgumentScalarReference(const T argument)
    {
        basicTuner->AddArgumentScalarReference(argument);
    }

    template void PUBLIC_API ExtendedTuner::addArgumentScalarReference<short>(const short argument);
    template void PUBLIC_API ExtendedTuner::addArgumentScalarReference<int>(const int argument);
    template void PUBLIC_API ExtendedTuner::addArgumentScalarReference<size_t>(const size_t argument);
    template void PUBLIC_API ExtendedTuner::addArgumentScalarReference<half>(const half argument);
    template void PUBLIC_API ExtendedTuner::addArgumentScalarReference<float>(const float argument);
    template void PUBLIC_API ExtendedTuner::addArgumentScalarReference<double>(const double argument);
    template void PUBLIC_API ExtendedTuner::addArgumentScalarReference<float2>(const float2 argument);
    template void PUBLIC_API ExtendedTuner::addArgumentScalarReference<double2>(const double2 argument);

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
