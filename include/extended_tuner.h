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

using UniqueTuner = std::unique_ptr<Tuner>;
using UniqueConfigurator = std::unique_ptr<TunerConfigurator>;

class ExtendedTuner
{
public:
    // Initializes the extended tuner by providing basic tuner (with already set arguments, parameters, etc.) and its configurator.
    explicit PUBLIC_API ExtendedTuner(UniqueTuner basicTuner, UniqueConfigurator configurator);
    
    // Extended tuner destructor.
    PUBLIC_API ~ExtendedTuner();

    // Runs single kernel using the provided configurator.
    PUBLIC_API void runSingleKernel(const size_t id, const ParameterRange &parameter_values);

    // Starts tuning process using the provided configurator.
    PUBLIC_API void tune();

private:
    UniqueTuner basicTuner;
    UniqueConfigurator configurator;
};

} // namespace cltune

#endif // CLTUNE_EXTENDED_TUNER_H_
