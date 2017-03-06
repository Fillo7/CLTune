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

    virtual PublicTunerResult customizedComputation(const ParameterRange& configuration, const IntRange& currentGlobal, const IntRange& currentLocal)
    {
        return tuner->RunSingleKernel(kernelId, configuration);
    }

private:
    Tuner* tuner;
    size_t kernelId;
};

} // namespace cltune

#endif // CLTUNE_DEFAULT_CONFIGURATOR_H_
