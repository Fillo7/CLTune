#ifndef CLTUNE_TUNER_CONFIGURATOR_H_
#define CLTUNE_TUNER_CONFIGURATOR_H_

#include "internal/internal_api.h"

namespace cltune
{

class TunerConfigurator
{
public:
    // Destructor implementation provided by default.
    virtual ~TunerConfigurator() {}

    // Method which is executed in extended tuner, provides finer control over kernel execution.
    // Needs to launch the kernel using specified configuration.
    // Needs to return tuning result for current kernel run, result is provided by built-in tuner method.
    // May compute additional data on CPU, which is later used during kernel execution.
    // May do additional operations on output provided by kernel execution.
    // Its running time is measured and added to kernel execution time.
    virtual PublicTunerResult customizedComputation(const ParameterRange& configuration) = 0;
};

} // namespace cltune

#endif // CLTUNE_TUNER_CONFIGURATOR_H_
