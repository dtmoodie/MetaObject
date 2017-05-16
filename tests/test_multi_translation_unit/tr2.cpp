#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include "MetaObject/params/MetaParam.hpp"
#include "MetaObject/params/buffers/CircularBuffer.hpp"
#include "MetaObject/params/buffers/StreamBuffer.hpp"
#include "MetaObject/params/buffers/Map.hpp"
#include "MetaObject/IMetaObject.hpp"
#include "MetaObject/signals/TSignal.hpp"
#include "MetaObject/detail/Counter.hpp"
#include "MetaObject/detail/MetaObjectMacros.hpp"
#include "MetaObject/signals/detail/SignalMacros.hpp"
#include "MetaObject/signals/detail/SlotMacros.hpp"
#include "MetaObject/params/ParamMacros.hpp"
#include "MetaObject/params/TParamPtr.hpp"
#include "MetaObject/params/TInputParam.hpp"
#include "MetaObject/logging/CompileLogger.hpp"
#include "MetaObject/params/buffers/BufferFactory.hpp"
#include "RuntimeObjectSystem/RuntimeObjectSystem.h"
#include "RuntimeObjectSystem/IObjectFactorySystem.h"


#ifdef _MSC_VER
#include <boost/test/unit_test.hpp>
#else
#define BOOST_TEST_MODULE "multi_translation_unit"
#include <boost/test/included/unit_test.hpp>
#endif
#include <boost/thread.hpp>
#include <iostream>

using namespace mo;

