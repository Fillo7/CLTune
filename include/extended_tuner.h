#ifndef CLTUNE_EXTENDED_TUNER_H_
#define CLTUNE_EXTENDED_TUNER_H_

#include <memory>

#include "cltune.h"
#include "tuner_configurator.h"

// Exports library functions under Windows when building a DLL.
// See also: https://msdn.microsoft.com/en-us/library/a90k134d.aspx
#ifdef _WIN32
#define PUBLIC_API __declspec(dllexport)
#else
#define PUBLIC_API
#endif

namespace cltune
{

using UniqueConfigurator = std::unique_ptr<TunerConfigurator>;

class ExtendedTuner
{
public:
    // Initializes the extended tuner by providing platform id and device id.
    explicit PUBLIC_API ExtendedTuner(size_t platformId, size_t deviceId);
    
    // Extended tuner destructor.
    PUBLIC_API ~ExtendedTuner();

    // Adds a new kernel to the list of tuning-kernels and returns a unique ID (to be used when adding tuning parameters).
    // Either loads the source from filenames or from string.
    size_t PUBLIC_API addKernel(const std::vector<std::string>& filenames, const std::string& kernelName, const IntRange& global,
                                const IntRange& local);
    size_t PUBLIC_API addKernelFromString(const std::string& source, const std::string& kernelName, const IntRange& global, const IntRange& local);

    // Sets the reference kernel. Same as the AddKernel function, but in this case there is only one reference kernel.
    // Calling this function again will overwrite the previous reference kernel.
    void PUBLIC_API setReference(const std::vector<std::string>& filenames, const std::string& kernelName, const IntRange& global,
                                 const IntRange& local);
    void PUBLIC_API setReferenceFromString(const std::string& source, const std::string& kernelName, const IntRange& global, const IntRange& local);

    // Adds a new tuning parameter for a kernel with a specific ID. The parameter has a name, the number of values, and a list of values.
    void PUBLIC_API addParameter(const size_t id, const std::string& parameterName, const std::initializer_list<size_t>& values);

    // As above, but now adds a single valued parameter to the reference
    void PUBLIC_API addParameterReference(const std::string& parameterName, const size_t value);

    // Modifies the global or local thread-size (integers) by one of the parameters (strings). The modifier can be multiplication or division.
    void PUBLIC_API mulGlobalSize(const size_t id, const StringRange range);
    void PUBLIC_API divGlobalSize(const size_t id, const StringRange range);
    void PUBLIC_API mulLocalSize(const size_t id, const StringRange range);
    void PUBLIC_API divLocalSize(const size_t id, const StringRange range);

    // Sets number of iterations that multirun kernel has to complete in order to produce complete result. Number of iterations is specified
    // by previously added tuning parameter, which allows it to become part of the tuning process. Single run kernels do not need to use this method.
    void PUBLIC_API setMultirunKernelIterations(const size_t id, const std::string& parameterName);

    // Adds a new constraint to the set of parameters (e.g. must be equal or larger than). The constraints come in the form of a function object
    // which takes a number of tuning parameters, given as a vector of strings (parameter names). Their names are later substituted by actual values.
    void PUBLIC_API addConstraint(const size_t id, ConstraintFunction validIf, const std::vector<std::string>& parameters);

    // As above, but for local memory usage. If this function is not called, it is assumed that the
    // local memory usage is 0: no configurations will be excluded because of too much local memory.
    void PUBLIC_API setLocalMemoryUsage(const size_t id, LocalMemoryFunction amount, const std::vector<std::string>& parameters);

    // Functions to add kernel-arguments for input buffers, output buffers, and scalars. Make sure to
    // call these in the order in which the arguments appear in the kernel.
    template <typename T> void addArgumentInput(const size_t id, const std::vector<T>& source);
    template <typename T> void addArgumentOutput(const size_t id, const std::vector<T>& source);
    template <typename T> void addArgumentScalar(const size_t id, const T argument);

    // Same as above, but for reference kernel.
    template <typename T> void addArgumentInputReference(const std::vector<T>& source);
    template <typename T> void addArgumentOutputReference(const std::vector<T>& source);
    template <typename T> void addArgumentScalarReference(const T argument);

    // Configures a specific search method. The default search method is "FullSearch". These are implemented as separate functions since
    // they each take a different number of arguments.
    void PUBLIC_API useFullSearch();
    void PUBLIC_API useRandomSearch(const double fraction);
    void PUBLIC_API useAnnealing(const double fraction, const double maxTemperature);
    void PUBLIC_API usePSO(const double fraction, const size_t swarmSize, const double influenceGlobal, const double influenceLocal,
                           const double influenceRandom);

    // Uses chosen technique for results comparison. Currently available techniques are absolute difference and side by side comparison.
    void PUBLIC_API chooseVerificationTechnique(const VerificationTechnique technique);
    void PUBLIC_API chooseVerificationTechnique(const VerificationTechnique technique, const double toleranceTreshold);

    // Outputs the search process to a file
    void PUBLIC_API outputSearchLog(const std::string& filename);

    // Trains a machine learning model based on the search space explored so far. Then, all the missing data-points are estimated
    // based on this model. This is only useful if a fraction of the search space is explored, as is the case when doing random-search.
    void PUBLIC_API modelPrediction(const Model modelType, const float validationFraction, const size_t testTopXConfigurations);

    // Prints the results of the tuning either to screen (stdout) or to a specific output-file. Returns the execution time in miliseconds.
    double PUBLIC_API printToScreen() const;
    void PUBLIC_API printFormatted() const;
    void PUBLIC_API printJSON(const std::string& filename, const std::vector<std::pair<std::string, std::string>>& descriptions) const;
    void PUBLIC_API printToFile(const std::string& filename) const;

    // Runs single kernel using the provided configurator.
    PUBLIC_API void runSingleKernel(const size_t id, const ParameterRange& parameterValues);

    // Starts tuning process using the provided configurator.
    PUBLIC_API void tune();

    // Sets the tuner configurator to specified custom configurator.
    PUBLIC_API void setConfigurator(UniqueConfigurator configurator);

private:
    std::unique_ptr<Tuner> basicTuner;
    UniqueConfigurator configurator;
};

} // namespace cltune

#endif // CLTUNE_EXTENDED_TUNER_H_
