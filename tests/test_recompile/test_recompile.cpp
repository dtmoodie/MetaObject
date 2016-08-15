#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include "MetaObject/Signals/detail/SignalMacros.hpp"
#include "MetaObject/Signals/detail/SlotMacros.hpp"
#include "MetaObject/IMetaObject.hpp"
#include "MetaObject/Signals/TypedSignal.hpp"
#include "MetaObject/Detail/Counter.hpp"
#include "MetaObject/Detail/MetaObjectMacros.hpp"
#include "MetaObject/Parameters//ParameterMacros.hpp"
#include "MetaObject/Parameters/TypedParameterPtr.hpp"
#include "MetaObject/Parameters/TypedInputParameter.hpp"
#include "RuntimeObjectSystem.h"
#include "IObjectFactorySystem.h"
#include "shared_ptr.hpp"

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "MetaObject"
#include <boost/test/unit_test.hpp>
#include <iostream>

using namespace mo;
const size_t LOGSYSTEM_MAX_BUFFER = 4096;
struct test_meta_object_signals: public IMetaObject
{
    ~test_meta_object_signals()
    {
        std::cout << "Deleting object\n";
    }
    MO_BEGIN(test_meta_object_signals);
    MO_SIGNAL(void, test_void);
    MO_SIGNAL(void, test_int, int);
    MO_END;
};

struct test_meta_object_slots: public IMetaObject
{
    MO_BEGIN(test_meta_object_slots);
    MO_SLOT(void, test_void);
    MO_SLOT(void, test_int, int);
    PROPERTY(int, call_count, 0);
    MO_END;
};

struct test_meta_object_parameters: public IMetaObject
{
    MO_BEGIN(test_meta_object_parameters);
    PARAM(int, test, 5);
    MO_END;
};

void test_meta_object_slots::test_void()
{
    ++call_count;
}

void test_meta_object_slots::test_int(int value)
{
    call_count += value;
}


MO_REGISTER_OBJECT(test_meta_object_signals);
MO_REGISTER_OBJECT(test_meta_object_slots);
MO_REGISTER_OBJECT(test_meta_object_parameters);

class build_callback: public ITestBuildNotifier
{
    virtual bool TestBuildCallback(const char* file, TestBuildResult type)
    {
        std::cout << "[" << file << "] - ";
        switch(type)
        {
        case TESTBUILDRRESULT_SUCCESS:
            std::cout << "TESTBUILDRRESULT_SUCCESS\n"; break;
        case TESTBUILDRRESULT_NO_FILES_TO_BUILD:
            std::cout << "TESTBUILDRRESULT_NO_FILES_TO_BUILD\n"; break;
        case TESTBUILDRRESULT_BUILD_FILE_GONE:
            std::cout << "TESTBUILDRRESULT_BUILD_FILE_GONE\n"; break;
        case TESTBUILDRRESULT_BUILD_NOT_STARTED:
            std::cout << "TESTBUILDRRESULT_BUILD_NOT_STARTED\n"; break;
        case TESTBUILDRRESULT_BUILD_FAILED:
            std::cout << "TESTBUILDRRESULT_BUILD_FAILED\n"; break;
        case TESTBUILDRRESULT_OBJECT_SWAP_FAIL:
            std::cout << "TESTBUILDRRESULT_OBJECT_SWAP_FAIL\n"; break;
        }
        return true;
    }
    virtual bool TestBuildWaitAndUpdate()
    {
        return true;
    }
};


class StdioLogSystem : public ICompilerLogger
{
public:    
    virtual void LogError(const char * format, ...)
    {
        va_list args;
        va_start(args, format);
        LogInternal(format, args);
    }
    virtual void LogWarning(const char * format, ...)
    {
        va_list args;
        va_start(args, format);
        LogInternal(format, args);
    }
    virtual void LogInfo(const char * format, ...)
    {
        va_list args;
        va_start(args, format);
        LogInternal(format, args);
    }

protected:
    void LogInternal(const char * format, va_list args)
    {
        vsnprintf(m_buff, LOGSYSTEM_MAX_BUFFER-1, format, args);
        // Make sure there's a limit to the amount of rubbish we can output
        m_buff[LOGSYSTEM_MAX_BUFFER-1] = '\0';

        std::cout << m_buff;
#ifdef _WIN32
        OutputDebugStringA( m_buff );
#endif
    }

    char m_buff[LOGSYSTEM_MAX_BUFFER];

};

build_callback* cb = nullptr;
BOOST_AUTO_TEST_CASE(test_recompile)
{
    cb = new build_callback;
    //MetaObjectFactory::Instance()->GetObjectSystem()->SetupObjectConstructors(PerModuleInterface::GetInstance());
	MetaObjectFactory::Instance()->RegisterTranslationUnit();
    BOOST_REQUIRE_EQUAL(MetaObjectFactory::Instance()->GetObjectSystem()->TestBuildAllRuntimeSourceFiles(cb, true), 0);
}

BOOST_AUTO_TEST_CASE(test_obj_swap)
{
    auto constructor = MetaObjectFactory::Instance()->GetObjectSystem()->GetObjectFactorySystem()->GetConstructor("test_meta_object_signals");
    auto obj = constructor->Construct();
    BOOST_REQUIRE(obj);
    auto state = constructor->GetState(obj->GetPerTypeId());
    BOOST_REQUIRE(state);
    auto ptr = state->GetSharedPtr();
    rcc::shared_ptr<test_meta_object_signals> typed_ptr(ptr);
    BOOST_REQUIRE(!typed_ptr.empty());
    BOOST_REQUIRE_EQUAL(MetaObjectFactory::Instance()->GetObjectSystem()->TestBuildAllRuntimeSourceFiles(cb, true), 0);
    BOOST_REQUIRE(!typed_ptr.empty());
    auto signals = typed_ptr->GetSignalInfo();
    BOOST_REQUIRE_EQUAL(signals.size(), 2);
}

BOOST_AUTO_TEST_CASE(test_pointer_mechanics)
{
    auto obj = test_meta_object_signals::Create();
    BOOST_REQUIRE(!obj.empty());
    BOOST_REQUIRE(obj.GetState());
    BOOST_REQUIRE_EQUAL(obj.GetState()->ObjectCount(), 1);
    BOOST_REQUIRE_EQUAL(obj.GetState()->StateCount(), 1);
    // Test construction of weak pointers from shared pointers
    {
        rcc::weak_ptr<test_meta_object_signals> weak_ptr(obj);
        BOOST_REQUIRE_EQUAL(obj.GetState()->ObjectCount(), 1);
        BOOST_REQUIRE_EQUAL(obj.GetState()->StateCount(), 2);
    }
    BOOST_REQUIRE_EQUAL(obj.GetState()->ObjectCount(), 1);
    BOOST_REQUIRE_EQUAL(obj.GetState()->StateCount(), 1);
    
    {
        rcc::weak_ptr<IMetaObject> weak_ptr_meta_object;
        weak_ptr_meta_object = rcc::weak_ptr<IMetaObject>(obj);
        BOOST_REQUIRE_EQUAL(obj.GetState()->ObjectCount(), 1);
        BOOST_REQUIRE_EQUAL(obj.GetState()->StateCount(), 2);
    }
    BOOST_REQUIRE_EQUAL(obj.GetState()->ObjectCount(), 1);
    BOOST_REQUIRE_EQUAL(obj.GetState()->StateCount(), 1);

    // Test construction of weak pointers from raw object pointer
    {
        rcc::weak_ptr<IMetaObject> weak_ptr_meta_object(obj.Get());
        BOOST_REQUIRE_EQUAL(obj.GetState()->ObjectCount(), 1);
        BOOST_REQUIRE_EQUAL(obj.GetState()->StateCount(), 2);
    }

    BOOST_REQUIRE_EQUAL(obj.GetState()->ObjectCount(), 1);
    BOOST_REQUIRE_EQUAL(obj.GetState()->StateCount(), 1);

    {
        rcc::weak_ptr<IMetaObject> weak_ptr_meta_object = rcc::weak_ptr<IMetaObject>(obj.Get());
        BOOST_REQUIRE_EQUAL(obj.GetState()->ObjectCount(), 1);
        BOOST_REQUIRE_EQUAL(obj.GetState()->StateCount(), 2);
    }

    BOOST_REQUIRE_EQUAL(obj.GetState()->ObjectCount(), 1);
    BOOST_REQUIRE_EQUAL(obj.GetState()->StateCount(), 1);

    // Test shared pointer mechanics
    {
        rcc::shared_ptr<test_meta_object_signals> shared(obj);
        BOOST_REQUIRE_EQUAL(obj.GetState()->ObjectCount(), 2);
        BOOST_REQUIRE_EQUAL(obj.GetState()->StateCount(), 2);
    }
    BOOST_REQUIRE_EQUAL(obj.GetState()->ObjectCount(), 1);
    BOOST_REQUIRE_EQUAL(obj.GetState()->StateCount(), 1);

    {
        rcc::shared_ptr<IMetaObject> shared_ptr_meta_object;
        shared_ptr_meta_object = rcc::shared_ptr<IMetaObject>(obj);
        BOOST_REQUIRE_EQUAL(obj.GetState()->ObjectCount(), 2);
        BOOST_REQUIRE_EQUAL(obj.GetState()->StateCount(), 2);
    }
    BOOST_REQUIRE_EQUAL(obj.GetState()->ObjectCount(), 1);
    BOOST_REQUIRE_EQUAL(obj.GetState()->StateCount(), 1);

    // Test construction of weak pointers from raw object pointer
    {
        rcc::shared_ptr<IMetaObject> shared_ptr_meta_object(obj.Get());
        BOOST_REQUIRE_EQUAL(obj.GetState()->ObjectCount(), 2);
        BOOST_REQUIRE_EQUAL(obj.GetState()->StateCount(), 2);
    }

    BOOST_REQUIRE_EQUAL(obj.GetState()->ObjectCount(), 1);
    BOOST_REQUIRE_EQUAL(obj.GetState()->StateCount(), 1);

    {
        rcc::shared_ptr<IMetaObject> shared_ptr_meta_object = rcc::shared_ptr<IMetaObject>(obj.Get());
        BOOST_REQUIRE_EQUAL(obj.GetState()->ObjectCount(), 2);
        BOOST_REQUIRE_EQUAL(obj.GetState()->StateCount(), 2);
    }

    BOOST_REQUIRE_EQUAL(obj.GetState()->ObjectCount(), 1);
    BOOST_REQUIRE_EQUAL(obj.GetState()->StateCount(), 1);

}

BOOST_AUTO_TEST_CASE(test_creation_function)
{
    auto obj = test_meta_object_signals::Create();
    BOOST_REQUIRE_EQUAL(MetaObjectFactory::Instance()->GetObjectSystem()->TestBuildAllRuntimeSourceFiles(cb, true), 0);
}

BOOST_AUTO_TEST_CASE(test_reconnect_signals)
{
    auto signals = test_meta_object_signals::Create();
    auto slots = test_meta_object_slots::Create();
    auto state = signals->GetConstructor()->GetState(signals->GetPerTypeId());
    IMetaObject::Connect(signals.Get(), "test_int", slots.Get(), "test_int");
    int value  = 5;
    signals->sig_test_int(value);
    BOOST_REQUIRE_EQUAL(slots->call_count, value);
    BOOST_REQUIRE_EQUAL(MetaObjectFactory::Instance()->GetObjectSystem()->TestBuildAllRuntimeSourceFiles(cb, true), 0);
    
    signals->sig_test_int(value);
    BOOST_REQUIRE_EQUAL(slots->call_count, 10);
}

BOOST_AUTO_TEST_CASE(test_parameter_persistence_recompile)
{
    auto obj = test_meta_object_parameters::Create();
    BOOST_REQUIRE_EQUAL(obj->test, 5);
    obj->test = 10;
    BOOST_REQUIRE_EQUAL(MetaObjectFactory::Instance()->GetObjectSystem()->TestBuildAllRuntimeSourceFiles(cb, true), 0);
    BOOST_REQUIRE_EQUAL(obj->test, 10);
}

BOOST_AUTO_TEST_CASE(test_object_cleanup)
{
    AUDynArray<IObjectConstructor*> constructors;
    MetaObjectFactory::Instance()->GetObjectSystem()->GetObjectFactorySystem()->GetAll(constructors);
    for(int i = 0; i < constructors.Size(); ++i)
    {
        BOOST_REQUIRE_EQUAL(constructors[i]->GetNumberConstructedObjects(), 0);
    }
    delete cb;
}