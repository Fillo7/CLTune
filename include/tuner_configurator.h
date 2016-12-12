#ifndef CLTUNE_TUNER_CONFIGURATOR_H_
#define CLTUNE_TUNER_CONFIGURATOR_H_

#include "cltune.h"

namespace cltune
{

class TunerConfigurator
{
public:
    // Destructor implementation provided by default.
    virtual ~TunerConfigurator() {}

    // Method which is executed in extended tuner to control tuning process and oversee kernel execution.
    // Needs to launch the kernel using specified configuration.
    // Needs to store tuning result from each kernel run, results are provided by respective tuner method.
    // May compute additional data on CPU, which is later used during kernel execution.
    // May do additional operations on output provided by kernel execution.
    // Its running time is measured and added to kernel execution time.
    virtual void customizedComputation() = 0;

    // Method which is executed in extended tuner to acquire and process tuning results.
    // Needs to return results acquired during customizedComputation() method execution.
    virtual std::vector<PublicTunerResult> getTuningResults() = 0;
};

} // namespace cltune

#endif // CLTUNE_TUNER_CONFIGURATOR_H_
