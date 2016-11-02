#ifndef CLTUNE_EXTENDED_TUNER_H_
#define CLTUNE_EXTENDED_TUNER_H_

#include "cltune.h"

// Exports library functions under Windows when building a DLL. See also:
// https://msdn.microsoft.com/en-us/library/a90k134d.aspx
#ifdef _WIN32
#define PUBLIC_API __declspec(dllexport)
#else
#define PUBLIC_API
#endif

namespace cltune
{

class ExtendedTuner
{
public:
    // Initializes the extended tuner either with platform 0 and device 0 or with a custom platform/device
    explicit PUBLIC_API ExtendedTuner();
    explicit PUBLIC_API ExtendedTuner(size_t platform_id, size_t device_id);
    ~ExtendedTuner();

private:
    std::unique_ptr<Tuner> basicTuner;
};

} // namespace cltune

#endif // CLTUNE_EXTENDED_TUNER_H_