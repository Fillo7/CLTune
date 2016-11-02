#include "extended_tuner.h"

namespace cltune
{
    ExtendedTuner::ExtendedTuner():
        basicTuner(new Tuner())
    {}
    
    ExtendedTuner::ExtendedTuner(size_t platform_id, size_t device_id):
        basicTuner(new Tuner(platform_id, device_id))
    {}

    ExtendedTuner::~ExtendedTuner() {}
} // namespace cltune