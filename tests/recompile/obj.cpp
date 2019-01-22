#include "obj.h"
#include <MetaObject/thread/fiber_include.hpp>

void MetaObjectSlots::test_void()
{
    ++call_count;
}

void MetaObjectSlots::test_int(int value)
{
    call_count += value;
}

MO_REGISTER_OBJECT(MetaObjectSignals);
MO_REGISTER_OBJECT(MetaObjectSlots);
MO_REGISTER_OBJECT(test_meta_object_parameters);
MO_REGISTER_OBJECT(test_meta_object_output);
MO_REGISTER_OBJECT(test_meta_object_input);
#ifdef HAVE_CUDA
MO_REGISTER_OBJECT(test_cuda_object);
#endif
