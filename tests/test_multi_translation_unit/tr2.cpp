#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include "MetaObject/Params/MetaParam.hpp"
#include "MetaObject/Params/Buffers/CircularBuffer.hpp"
#include "MetaObject/Params/Buffers/StreamBuffer.hpp"
#include "MetaObject/Params/Buffers/Map.hpp"
#include "MetaObject/IMetaObject.hpp"
#include "MetaObject/Signals/TSignal.hpp"
#include "MetaObject/Detail/Counter.hpp"
#include "MetaObject/Detail/MetaObjectMacros.hpp"
#include "MetaObject/Signals/Detail/SignalMacros.hpp"
#include "MetaObject/Signals/Detail/SlotMacros.hpp"
#include "MetaObject/Params/ParamMacros.hpp"
#include "MetaObject/Params/TParamPtr.hpp"
#include "MetaObject/Params/TInputParam.hpp"
#include "MetaObject/Logging/CompileLogger.hpp"
#include "MetaObject/Params/Buffers/BufferFactory.hpp"
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

