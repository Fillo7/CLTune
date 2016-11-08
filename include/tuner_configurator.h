#ifndef CLTUNE_TUNER_CONFIGURATOR_H_
#define CLTUNE_TUNER_CONFIGURATOR_H_

namespace cltune
{

class TunerConfigurator
{
public:
    virtual ~TunerConfigurator() {}

    virtual void beforeTuning() = 0;
    virtual void afterTuning() = 0;
};

} // namespace cltune

#endif // CLTUNE_TUNER_CONFIGURATOR_H_
