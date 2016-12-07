#include <chrono>
#include <complex>
#include <iostream>
#include <fstream>

#include "CL/cl.h"
#include "extended_tuner.h"

namespace cltune
{
    using half = cl_half;
    using float2 = std::complex<float>;
    using double2 = std::complex<double>;

    // ==============================================================================================================================================
    // Constructors and destructor

    ExtendedTuner::ExtendedTuner(size_t platformId, size_t deviceId):
        kernelCount(0),
        basicTuner(new Tuner(platformId, deviceId))
    {}

    ExtendedTuner::~ExtendedTuner() {}

    // ==============================================================================================================================================
    // Kernel addition methods

    size_t ExtendedTuner::addKernel(const std::vector<std::string>& filenames, const std::string& kernelName, const IntRange& global,
                                    const IntRange& local)
    {
        size_t id = basicTuner->AddKernel(filenames, kernelName, global, local);
        kernelCount++; // Increase kernel count only after successful kernel addition
        return id;
    }

    size_t ExtendedTuner::addKernelFromString(const std::string& source, const std::string& kernelName, const IntRange& global,
                                              const IntRange& local)
    {
        size_t id = basicTuner->AddKernelFromString(source, kernelName, global, local);
        kernelCount++; // Increase kernel count only after successful kernel addition
        return id;
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

    // ==============================================================================================================================================
    // Tuning parameter addition methods

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

    // ==============================================================================================================================================
    // Argument addition methods

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

    // ==============================================================================================================================================
    // Additional settings methods

    void ExtendedTuner::useFullSearch(const size_t id)
    {
        basicTuner->UseFullSearch(id);
    }

    void ExtendedTuner::useRandomSearch(const size_t id, const double fraction)
    {
        basicTuner->UseRandomSearch(id, fraction);
    }

    void ExtendedTuner::useAnnealing(const size_t id, const double fraction, const double maxTemperature)
    {
        basicTuner->UseAnnealing(id, fraction, maxTemperature);
    }

    void ExtendedTuner::usePSO(const size_t id, const double fraction, const size_t swarmSize, const double influenceGlobal,
                               const double influenceLocal, const double influenceRandom)
    {
        basicTuner->UsePSO(id, fraction, swarmSize, influenceGlobal, influenceLocal, influenceRandom);
    }

    void ExtendedTuner::setConfigurator(const size_t id, UniqueConfigurator configurator)
    {
        size_t configuratorId = getConfiguratorIndex(id);

        if (configuratorId != -1)
        {
            configurators.at(configuratorId).second = std::move(configurator); // Original object is destroyed
            return;
        }

        configurators.push_back(std::make_pair(id, std::move(configurator)));
    }

    void ExtendedTuner::chooseVerificationMethod(const VerificationMethod method, const double toleranceTreshold)
    {
        basicTuner->ChooseVerificationMethod(method, toleranceTreshold);
    }

    void ExtendedTuner::outputSearchLog(const std::string& filename)
    {
        basicTuner->OutputSearchLog(filename);
    }

    void ExtendedTuner::modelPrediction(const Model modelType, const float validationFraction, const size_t testTopXConfigurations)
    {
        basicTuner->ModelPrediction(modelType, validationFraction, testTopXConfigurations);
    }

    // ==============================================================================================================================================
    // Tuning methods

    void ExtendedTuner::runSingleKernel(const size_t id, const ParameterRange& parameterValues)
    {
        /*size_t configuratorIndex = getConfiguratorIndex(id);

        auto beforeTuningBegin = std::chrono::high_resolution_clock::now();
        if (configuratorIndex >= 0)
        {
            configurators.at(configuratorIndex).second->beforeTuning();
        }
        auto beforeTuningEnd = std::chrono::high_resolution_clock::now();
        auto beforeDuration = std::chrono::duration_cast<std::chrono::milliseconds>(beforeTuningEnd - beforeTuningBegin).count();*/

        cltune::PublicTunerResult result = basicTuner->RunSingleKernel(id, parameterValues);

        /*auto afterTuningBegin = std::chrono::high_resolution_clock::now();
        if (configuratorIndex >= 0)
        {
            configurators.at(configuratorIndex).second->afterTuning();
        }
        auto afterTuningEnd = std::chrono::high_resolution_clock::now();
        auto afterDuration = std::chrono::duration_cast<std::chrono::milliseconds>(afterTuningEnd - afterTuningBegin).count();*/

        storeTunerResult(id, result);
        /*if (configuratorIndex >= 0)
        {
            storeTunerResult(id, result, (float)beforeDuration, (float)afterDuration);
        }
        else
        {
            storeTunerResult(id, result);
        }*/
    }

    void ExtendedTuner::tuneSingleKernel(const size_t id)
    {
        /*size_t configuratorIndex = getConfiguratorIndex(id);

        auto beforeTuningBegin = std::chrono::high_resolution_clock::now();
        if (configuratorIndex >= 0)
        {
            configurators.at(configuratorIndex).second->beforeTuning();
        }
        auto beforeTuningEnd = std::chrono::high_resolution_clock::now();
        auto beforeDuration = std::chrono::duration_cast<std::chrono::milliseconds>(beforeTuningEnd - beforeTuningBegin).count();*/

        std::vector<cltune::PublicTunerResult> results = basicTuner->TuneSingleKernel(id);

        /*auto afterTuningBegin = std::chrono::high_resolution_clock::now();
        if (configuratorIndex >= 0)
        {
            configurators.at(configuratorIndex).second->afterTuning();
        }
        auto afterTuningEnd = std::chrono::high_resolution_clock::now();
        auto afterDuration = std::chrono::duration_cast<std::chrono::milliseconds>(afterTuningEnd - afterTuningBegin).count();*/

        for (auto& result : results)
        {
            storeTunerResult(id, result);
            /*if (configuratorIndex >= 0)
            {
                storeTunerResult(id, result, (float)beforeDuration, (float)afterDuration);
            }
            else
            {
                storeTunerResult(id, result);
            }*/
        }
    }

    void ExtendedTuner::tuneSingleKernelCustomized(const size_t id)
    {
        size_t configuratorIndex = getConfiguratorIndex(id);

        if (configuratorIndex < 0)
        {
            throw std::runtime_error("Specified kernel has no configurator.");
        }

        auto begin = std::chrono::high_resolution_clock::now();

        configurators.at(configuratorIndex).second->customizedComputation();

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    }

    void ExtendedTuner::tuneAllKernels()
    {
        for (size_t id = 0; id < kernelCount; id++)
        {
            tuneSingleKernel(id);
        }
    }

    // ==============================================================================================================================================
    // Output methods

    void ExtendedTuner::printToScreen(const size_t id) const
    {
        std::cout << extHeader << extPrintingResultsToScreen << id << std::endl;
        printResults(id, std::cout);
    }

    void ExtendedTuner::printToScreen() const
    {
        for (size_t id = 0; id < kernelCount; id++)
        {
            printToScreen(id);
        }
    }

    void ExtendedTuner::printToFile(const size_t id, const std::string& filename) const
    {
        std::ofstream outputFile(filename);

        if (!outputFile.is_open())
        {
            std::cerr << extHeader << extNoFileOpen << std::endl;
            return;
        }
        std::cout << extHeader << extPrintingResultsToFile << id << std::endl;
        
        printResults(id, outputFile);

        outputFile.close();
    }

    void ExtendedTuner::printToFile(const std::string& filename) const
    {
        for (size_t id = 0; id < kernelCount; id++)
        {
            printToFile(id, filename);
        }
    }

    // ==============================================================================================================================================
    // Private methods

    size_t ExtendedTuner::getConfiguratorIndex(const size_t kernelId) const
    {
        for (size_t i = 0; i < configurators.size(); i++)
        {
            if (configurators.at(i).first == kernelId)
            {
                return i;
            }
        }

        return -1;
    }

    void ExtendedTuner::printKernelInfo(const cltune::PublicTunerResult& result, std::ostream& out) const
    {
        out << result.kernel_name << " " << result.threads << " ";
        for (auto& param : result.parameter_values)
        {
            out << "[" << param.first << ": " << param.second << "] ";
        }
        out << std::endl;
    }

    void ExtendedTuner::printResults(const size_t id, std::ostream& out) const
    {
        if (results.size() < 1)
        {
            out << extNoResults << std::endl;
            return;
        }

        ExtendedTunerResult best;
        best.basicResult.time = std::numeric_limits<float>::max();
        for (auto& result : results)
        {
            if (result.first == id && result.second.basicResult.status)
            {
                out << extKernelDuration << result.second.basicResult.time << extMs << std::endl;
                printKernelInfo(result.second.basicResult, out);

                if (best.basicResult.time > result.second.basicResult.time)
                {
                    // Copy all attributes, only the relevant ones will be used
                    best.hasConfigurator = result.second.hasConfigurator;
                    best.beforeDuration = result.second.beforeDuration;
                    best.afterDuration = result.second.afterDuration;
                    best.basicResult = result.second.basicResult;
                }
            }
        }

        out << std::endl << extFastestKernelDuration << best.basicResult.time << extMs << std::endl;
        printKernelInfo(best.basicResult, out);

        if (best.hasConfigurator)
        {
            out << extBeforeDuration << best.beforeDuration << extMs << std::endl;
            out << extAfterDuration << best.afterDuration << extMs << std::endl;
            out << extTotalDuration << best.beforeDuration + best.afterDuration + best.basicResult.time << extMs << std::endl;
        }
    }

    void ExtendedTuner::storeTunerResult(const size_t id, const cltune::PublicTunerResult& result)
    {
        ExtendedTunerResult extendedResult;
        extendedResult.basicResult = result;
        extendedResult.hasConfigurator = false;

        results.push_back(std::make_pair(id, extendedResult));
    }

    void ExtendedTuner::storeTunerResult(const size_t id, const cltune::PublicTunerResult& result,
                                         const float beforeDuration, const float afterDuration)
    {
        ExtendedTunerResult extendedResult;
        extendedResult.basicResult = result;
        extendedResult.hasConfigurator = true;
        extendedResult.beforeDuration = beforeDuration;
        extendedResult.afterDuration = afterDuration;

        results.push_back(std::make_pair(id, extendedResult));
    }

} // namespace cltune
