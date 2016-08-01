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


#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "MetaObject"
#include <boost/test/unit_test.hpp>
#include <iostream>

using namespace mo;
const size_t LOGSYSTEM_MAX_BUFFER = 4096;
struct test_meta_object_signals: public IMetaObject
{
    MO_BEGIN(test_meta_object_signals);
    MO_SIGNAL(void, test_void);
    MO_SIGNAL(void, test_int, int);
    MO_END;
};

MO_REGISTER_OBJECT(test_meta_object_signals)


BOOST_AUTO_TEST_CASE(test_recompile)
{
    class build_callback: public ITestBuildNotifier
    {
        virtual bool TestBuildCallback(const char* file, TestBuildResult type)
        {
            std::cout << "[" << file << "] failed!\n";
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
    RuntimeObjectSystem obj_sys;
    StdioLogSystem logger;
    obj_sys.Initialise(&logger, nullptr);
    build_callback cb;
    
    BOOST_REQUIRE_EQUAL(obj_sys.TestBuildAllRuntimeSourceFiles(&cb, true), 0);
}

BOOST_AUTO_TEST_CASE(test_obj_swap)
{
    RuntimeObjectSystem obj_sys;
    obj_sys.Initialise(nullptr, nullptr);
    auto constructor = obj_sys.GetObjectFactorySystem()->GetConstructor("test_meta_object_signals");
    BOOST_REQUIRE(constructor);
    auto obj = constructor->Construct();
    auto state = constructor->GetState(obj->GetPerTypeId());
    BOOST_REQUIRE(state);
    auto ptr = state->GetSharedPtr();
    rcc::shared_ptr<test_meta_object_signals> typed_ptr(ptr);
    BOOST_REQUIRE_EQUAL(obj_sys.TestBuildAllRuntimeSourceFiles(nullptr, true), 0);
}


