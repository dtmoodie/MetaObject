#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN

#include "RuntimeObjectSystem/RuntimeObjectSystem.h"
#include "RuntimeObjectSystem/IObjectFactorySystem.h"
#include "RuntimeObjectSystem/shared_ptr.hpp"
#include "obj.h"

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "MetaObject"
#if WIN32
#include <boost/test/unit_test.hpp>
#else
#include <boost/test/included/unit_test.hpp>
#endif
#include <boost/filesystem.hpp>
#include <iostream>

using namespace mo;

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


build_callback* cb = nullptr;
BOOST_AUTO_TEST_CASE(test_recompile)
{
    cb = new build_callback;
    LOG(info) << "Current working directory " << boost::filesystem::current_path().string();
    MetaObjectFactory::instance()->registerTranslationUnit();
    BOOST_REQUIRE_EQUAL(MetaObjectFactory::instance()->getObjectSystem()->TestBuildAllRuntimeSourceFiles(cb, true), 0);
}

BOOST_AUTO_TEST_CASE(test_obj_swap)
{
    auto constructor = MetaObjectFactory::instance()->getObjectSystem()->GetObjectFactorySystem()->GetConstructor("test_meta_object_signals");
    auto obj = constructor->Construct();
    BOOST_REQUIRE(obj);
    auto state = constructor->GetState(obj->GetPerTypeId());
    BOOST_REQUIRE(state);
    auto ptr = state->GetSharedPtr();
    rcc::shared_ptr<test_meta_object_signals> T_ptr(ptr);
    BOOST_REQUIRE(!T_ptr.empty());
    BOOST_REQUIRE_EQUAL(MetaObjectFactory::instance()->getObjectSystem()->TestBuildAllRuntimeSourceFiles(cb, true), 0);
    BOOST_REQUIRE(!T_ptr.empty());
    std::vector<SignalInfo*> signals;
    T_ptr->getSignalInfo(signals);
    BOOST_REQUIRE_EQUAL(signals.size(), 2);
}

BOOST_AUTO_TEST_CASE(test_pointer_mechanics)
{
    auto obj = test_meta_object_signals::create();
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
        rcc::weak_ptr<IMetaObject> weak_ptr_meta_object(obj.get());
        BOOST_REQUIRE_EQUAL(obj.GetState()->ObjectCount(), 1);
        BOOST_REQUIRE_EQUAL(obj.GetState()->StateCount(), 2);
    }

    BOOST_REQUIRE_EQUAL(obj.GetState()->ObjectCount(), 1);
    BOOST_REQUIRE_EQUAL(obj.GetState()->StateCount(), 1);

    {
        rcc::weak_ptr<IMetaObject> weak_ptr_meta_object = rcc::weak_ptr<IMetaObject>(obj.get());
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
        rcc::shared_ptr<IMetaObject> shared_ptr_meta_object(obj.get());
        BOOST_REQUIRE_EQUAL(obj.GetState()->ObjectCount(), 2);
        BOOST_REQUIRE_EQUAL(obj.GetState()->StateCount(), 2);
    }

    BOOST_REQUIRE_EQUAL(obj.GetState()->ObjectCount(), 1);
    BOOST_REQUIRE_EQUAL(obj.GetState()->StateCount(), 1);

    {
        rcc::shared_ptr<IMetaObject> shared_ptr_meta_object = rcc::shared_ptr<IMetaObject>(obj.get());
        BOOST_REQUIRE_EQUAL(obj.GetState()->ObjectCount(), 2);
        BOOST_REQUIRE_EQUAL(obj.GetState()->StateCount(), 2);
    }

    BOOST_REQUIRE_EQUAL(obj.GetState()->ObjectCount(), 1);
    BOOST_REQUIRE_EQUAL(obj.GetState()->StateCount(), 1);

}

BOOST_AUTO_TEST_CASE(test_creation_function)
{
    auto obj = test_meta_object_signals::create();
    BOOST_REQUIRE_EQUAL(MetaObjectFactory::instance()->getObjectSystem()->TestBuildAllRuntimeSourceFiles(cb, true), 0);
}

BOOST_AUTO_TEST_CASE(test_reConnect_signals)
{
    auto signals = test_meta_object_signals::create();
    auto slots = test_meta_object_slots::create();
    //auto state = signals->getConstructor()->GetState(signals->GetPerTypeId());
    IMetaObject::connect(signals.get(), "test_int", slots.get(), "test_int");
    int value  = 5;
    signals->sig_test_int(value);
    BOOST_REQUIRE_EQUAL(slots->call_count, value);
    BOOST_REQUIRE_EQUAL(MetaObjectFactory::instance()->getObjectSystem()->TestBuildAllRuntimeSourceFiles(cb, true), 0);

    signals->sig_test_int(value);
    BOOST_REQUIRE_EQUAL(slots->call_count, 10);
}

BOOST_AUTO_TEST_CASE(test_input_output_Param)
{
    auto output = rcc::shared_ptr<test_meta_object_output>::create();
    auto input = rcc::shared_ptr<test_meta_object_input>::create();
    auto output_param = output->getOutput("test_output");
    BOOST_REQUIRE(output_param);
    auto input_param = input->getInput("test_input");
    BOOST_REQUIRE(input_param);

    BOOST_REQUIRE(IMetaObject::connectInput(output.get(), output_param, input.get(), input_param));
    output->test_output = 5;
    BOOST_REQUIRE(input->test_input);
    BOOST_REQUIRE_EQUAL(*input->test_input, output->test_output);
    BOOST_REQUIRE_EQUAL(MetaObjectFactory::instance()->getObjectSystem()->TestBuildAllRuntimeSourceFiles(cb, true), 0);
    output->test_output = 10;
    BOOST_REQUIRE(input->test_input);
    BOOST_REQUIRE_EQUAL(*input->test_input, output->test_output);
    BOOST_REQUIRE_EQUAL(*input->test_input, 10);
}

BOOST_AUTO_TEST_CASE(test_Param_persistence_recompile)
{
    auto obj = test_meta_object_parameters::create();
    BOOST_REQUIRE_EQUAL(obj->test, 5);
    obj->test = 10;
    BOOST_REQUIRE_EQUAL(MetaObjectFactory::instance()->getObjectSystem()->TestBuildAllRuntimeSourceFiles(cb, true), 0);
    BOOST_REQUIRE_EQUAL(obj->test, 10);
}



BOOST_AUTO_TEST_CASE(test_multiple_objects)
{
    auto obj1 = rcc::shared_ptr<test_meta_object_parameters>::create();
    auto obj2 = rcc::shared_ptr<test_meta_object_parameters>::create();
    obj1->test = 1;
    obj2->test = 2;
    BOOST_REQUIRE_EQUAL(MetaObjectFactory::instance()->getObjectSystem()->TestBuildAllRuntimeSourceFiles(cb, true), 0);
    BOOST_REQUIRE_EQUAL(obj1->test, 1);
    BOOST_REQUIRE_EQUAL(obj2->test, 2);
}

#ifdef HAVE_CUDA
BOOST_AUTO_TEST_CASE(test_cuda_recompile)
{
    auto obj = rcc::shared_ptr<test_cuda_object>::create();
    obj->run_kernel();
    BOOST_REQUIRE_EQUAL(MetaObjectFactory::instance()->getObjectSystem()->TestBuildAllRuntimeSourceFiles(cb, true), 0);
    obj->run_kernel();
}
#endif



BOOST_AUTO_TEST_CASE(test_object_cleanup)
{
    AUDynArray<IObjectConstructor*> constructors;
    MetaObjectFactory::instance()->getObjectSystem()->GetObjectFactorySystem()->GetAll(constructors);
    for(int i = 0; i < constructors.Size(); ++i)
    {
        BOOST_REQUIRE_EQUAL(constructors[i]->GetNumberConstructedObjects(), 0);
    }
    delete cb;
}
