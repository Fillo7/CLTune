#ifndef CLTUNE_TUNER_CONFIGURATOR_H_
#define CLTUNE_TUNER_CONFIGURATOR_H_

namespace cltune
{

class TunerConfigurator
{
public:
    // Destructor implementation provided by default.
    virtual ~TunerConfigurator() {}

    // Method which is executed in extended tuner before tuning process.
    // May compute additional data on CPU, which is later used during kernel execution.
    // Its running time is measured and added to kernel execution time.
    virtual void beforeTuning() = 0;

    // Method which is executed in extended tuner after tuning process.
    // May use data provided by kernel and do some additional computations on CPU to modify output.
    // Its running time is measured and added to kernel execution time.
    virtual void afterTuning() = 0;
};

} // namespace cltune

#endif // CLTUNE_TUNER_CONFIGURATOR_H_
