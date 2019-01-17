// clang-format off
#include "MetaObject/core/detail/Counter.hpp"
#include "MetaObject/core.hpp"
#include "MetaObject/logging/CompileLogger.hpp"

#include <MetaObject/params/MetaParam.hpp>
#include <MetaObject/params/ParamMacros.hpp>
#include <MetaObject/params/TInputParam.hpp>
#include <MetaObject/params/TParamPtr.hpp>
#include <MetaObject/params/buffers/BufferFactory.hpp>
#include <MetaObject/params/detail/MetaParamImpl.hpp>

#include <MetaObject/object.hpp>
#include <MetaObject/serialization/memory.hpp>

#include "MetaObject/signals/TSignal.hpp"
#include "MetaObject/signals/detail/SignalMacros.hpp"
#include "MetaObject/signals/detail/SlotMacros.hpp"

#include "RuntimeObjectSystem/IObjectFactorySystem.h"
#include "RuntimeObjectSystem/RuntimeObjectSystem.h"
#include "RuntimeObjectSystem/shared_ptr.hpp"

#include "cereal/archives/portable_binary.hpp"
#include "cereal/archives/xml.hpp"

#include <MetaObject/types/file_types.hpp>
#include <MetaObject/runtime_reflection/visitor_traits/vector.hpp>


#include <fstream>
#include <istream>
// clang-format on
#ifdef HAVE_OPENCV
#include <opencv2/core.hpp>
#endif

#include <boost/test/test_tools.hpp>
#include <boost/test/auto_unit_test.hpp>

#include <boost/thread.hpp>
#include <iostream>

using namespace mo;

struct SerializableObject : public MetaObject
{
    ~SerializableObject() override;
    MO_BEGIN(SerializableObject)
       PARAM(int, test, 5)
       PARAM(int, test2, 6)
    MO_END
};

SerializableObject::~SerializableObject()
{
}

MO_REGISTER_OBJECT(SerializableObject);
