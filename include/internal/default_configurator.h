#ifndef CLTUNE_DEFAULT_CONFIGURATOR_H_
#define CLTUNE_DEFAULT_CONFIGURATOR_H_

#include "tuner_configurator.h"
#include "internal/internal_api.h"

namespace cltune
{

class DefaultConfigurator: public TunerConfigurator
{
public:
    DefaultConfigurator(Tuner& tuner, size_t kernelId):
        kernelId(kernelId)
    {
        this->tuner = &tuner;
    }

    virtual void customizedComputation()
    {
        size_t configurationsCount = tuner->GetNumConfigurations(kernelId); // Initialize searcher and get total number of unique configurations

        for (size_t i = 0; i < configurationsCount; i++)
        {
            ParameterRange configuration = tuner->GetNextConfiguration(kernelId); // Acquire next configuration
            PublicTunerResult result = tuner->RunSingleKernel(kernelId, configuration); // Run kernel with acquired configuration
            
            // Notify tuner of previous kernel running time, this is needed for computing next configuration
            tuner->UpdateKernelConfiguration(kernelId, result.time); 

            // Store result of current kernel run
            tuningResults.push_back(result);
        }
    }

    virtual std::vector<PublicTunerResult> getTuningResults()
    {
        return tuningResults;
    }

private:
    Tuner* tuner;
    size_t kernelId;
    std::vector<PublicTunerResult> tuningResults;
};

} // namespace cltune

#endif // CLTUNE_DEFAULT_CONFIGURATOR_H_
