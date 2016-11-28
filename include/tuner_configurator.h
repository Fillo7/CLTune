#ifndef CLTUNE_TUNER_CONFIGURATOR_H_
#define CLTUNE_TUNER_CONFIGURATOR_H_

namespace cltune
{

class TunerConfigurator
{
public:
    // Destructor implementation provided by default.
    virtual ~TunerConfigurator() {}

    // Method which is executed in extended tuner to control tuning process and oversee kernel execution.
    // May compute additional data on CPU, which is later used during kernel execution.
    // May do additional operations on output provided by kernel execution.
    // Its running time is measured and added to kernel execution time.
    virtual void extendedComputation() = 0;
};

} // namespace cltune

#endif // CLTUNE_TUNER_CONFIGURATOR_H_
