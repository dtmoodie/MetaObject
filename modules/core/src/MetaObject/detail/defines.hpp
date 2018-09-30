#include "MetaObject/core/metaobject_config.hpp"

#if MO_HAVE_CUDA
#include "host_defines.h"

#ifdef _MSC_VER
#define MO_INLINE __forceinline
#else
#define MO_INLINE inline __attribute__((always_inline))
#endif

#ifdef _MSC_VER
#define MO_XINLINE __device__ __host__ __forceinline
#else
#define MO_XINLINE __device__ __host__ inline __attribute__((always_inline))
#endif

#else

#ifdef _MSC_VER
#define MO_INLINE __forceinline
#else
#define MO_INLINE inline __attribute__((always_inline))
#endif

#endif
