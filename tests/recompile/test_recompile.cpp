#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN

#include "RuntimeObjectSystem/IObjectFactorySystem.h"
#include "RuntimeObjectSystem/RuntimeObjectSystem.h"
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

class build_callback : public ITestBuildNotifier
{
    virtual bool TestBuildCallback(const char* file, TestBuildResult type)
    {
        std::cout << "[" << file << "] - ";
        switch (type)
        {
        case TESTBUILDRRESULT_SUCCESS:
            std::cout << "TESTBUILDRRESULT_SUCCESS\n";
            break;
        case TESTBUILDRRESULT_NO_FILES_TO_BUILD:
            std::cout << "TESTBUILDRRESULT_NO_FILES_TO_BUILD\n";
            break;
        case TESTBUILDRRESULT_BUILD_FILE_GONE:
            std::cout << "TESTBUILDRRESULT_BUILD_FILE_GONE\n";
            break;
        case TESTBUILDRRESULT_BUILD_NOT_STARTED:
            std::cout << "TESTBUILDRRESULT_BUILD_NOT_STARTED\n";
            break;
        case TESTBUILDRRESULT_BUILD_FAILED:
            std::cout << "TESTBUILDRRESULT_BUILD_FAILED\n";
            break;
        case TESTBUILDRRESULT_OBJECT_SWAP_FAIL:
            std::cout << "TESTBUILDRRESULT_OBJECT_SWAP_FAIL\n";
            break;
        }
        return true;
    }
    virtual bool TestBuildWaitAndUpdate() { return true; }
};

build_callback* cb = nullptr;
BOOST_AUTO_TEST_CASE(test_recompile)
{
    cb = new build_callback;
    MO_LOG(info) << "Current working directory " << boost::filesystem::current_path().string();
    MetaObjectFactory::instance().registerTranslationUnit();
#ifdef _MSC_VER
    BOOST_REQUIRE(MetaObjectFactory::instance().loadPlugin("./test_recompile_objectd.dll"));
#else
    BOOST_REQUIRE(MetaObjectFactory::instance().loadPlugin("./libtest_recompile_objectd.so"));
#endif

    BOOST_REQUIRE_EQUAL(MetaObjectFactory::instance().getObjectSystem()->TestBuildAllRuntimeSourceFiles(cb, true), 0);
}

BOOST_AUTO_TEST_CASE(test_obj_swap)
{
    auto constructor = MetaObjectFactory::instance().getObjectSystem()->GetObjectFactorySystem()->GetConstructor(
        "test_meta_object_signals");
    auto obj = constructor->Construct();
    BOOST_REQUIRE(obj);
    auto state = constructor->GetState(obj->GetPerTypeId());
    BOOST_REQUIRE(state);
    auto ptr = state->GetSharedPtr();
    rcc::shared_ptr<test_meta_object_signals> T_ptr(ptr);
    BOOST_REQUIRE(!T_ptr.empty());
    BOOST_REQUIRE_EQUAL(MetaObjectFactory::instance().getObjectSystem()->TestBuildAllRuntimeSourceFiles(cb, true), 0);
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
    BOOST_REQUIRE_EQUAL(MetaObjectFactory::instance().getObjectSystem()->TestBuildAllRuntimeSourceFiles(cb, true), 0);
}

BOOST_AUTO_TEST_CASE(test_reconnect_signals)
{
    auto emitter = test_meta_object_signals::create();
    auto receiver = test_meta_object_slots::create();
    // auto state = signals->getConstructor()->GetState(signals->GetPerTypeId());
    IMetaObject::connect(emitter.get(), "test_int", receiver.get(), "test_int");
    int value = 5;
    emitter->sig_test_int(value);
    BOOST_REQUIRE_EQUAL(receiver->call_count, value);
    BOOST_REQUIRE_EQUAL(MetaObjectFactory::instance().getObjectSystem()->TestBuildAllRuntimeSourceFiles(cb, true), 0);

    emitter->sig_test_int(value);
    BOOST_REQUIRE_EQUAL(receiver->call_count, 10);

    emitter->sig_test_int(value);
    BOOST_REQUIRE_EQUAL(receiver->call_count, 15);
}

BOOST_AUTO_TEST_CASE(test_input_output_param)
{
    auto ctx = mo::Context::create("test_input_output_param");
    auto output = rcc::shared_ptr<test_meta_object_output>::create();
    auto input = rcc::shared_ptr<test_meta_object_input>::create();
    input->setContext(ctx);
    output->setContext(ctx);
    auto output_param = output->getOutput("test_output");
    BOOST_REQUIRE(output_param);
    auto input_param = input->getInput("test_input");
    BOOST_REQUIRE(input_param);
    BOOST_REQUIRE_EQUAL(input->param_update_call_count, 0);
    BOOST_REQUIRE_EQUAL(output->param_update_call_count, 0);
    BOOST_REQUIRE(IMetaObject::connectInput(output.get(), output_param, input.get(), input_param));
    BOOST_REQUIRE_EQUAL(input->param_update_call_count, 1);
    BOOST_REQUIRE_EQUAL(output->param_update_call_count, 0);
    output->test_output_param.updateData(5);
    BOOST_REQUIRE_EQUAL(output->param_update_call_count, 1);
    BOOST_REQUIRE_EQUAL(input->param_update_call_count, 2);
    BOOST_REQUIRE(input->test_input);
    BOOST_REQUIRE(input->test_input_param.getInput(mo::OptionalTime()));
    BOOST_REQUIRE_EQUAL(*input->test_input, output->test_output);
    BOOST_REQUIRE_EQUAL(MetaObjectFactory::instance().getObjectSystem()->TestBuildAllRuntimeSourceFiles(cb, true), 0);
    BOOST_REQUIRE_EQUAL(input->param_update_call_count, 1);
    output->test_output_param.updateData(10);
    BOOST_REQUIRE_EQUAL(input->param_update_call_count, 2);
    BOOST_REQUIRE(input->test_input_param.getInput(mo::OptionalTime()));
    BOOST_REQUIRE(input->test_input);
    BOOST_REQUIRE_EQUAL(*input->test_input, output->test_output);
    BOOST_REQUIRE_EQUAL(*input->test_input, 10);
}

BOOST_AUTO_TEST_CASE(test_param_persistence_recompile)
{
    auto obj = test_meta_object_parameters::create();
    BOOST_REQUIRE_EQUAL(obj->test, 5);
    obj->test = 10;
    BOOST_REQUIRE_EQUAL(MetaObjectFactory::instance().getObjectSystem()->TestBuildAllRuntimeSourceFiles(cb, true), 0);
    BOOST_REQUIRE_EQUAL(obj->test, 10);
}

BOOST_AUTO_TEST_CASE(test_multiple_objects)
{
    auto obj1 = rcc::shared_ptr<test_meta_object_parameters>::create();
    auto obj2 = rcc::shared_ptr<test_meta_object_parameters>::create();
    obj1->test = 1;
    obj2->test = 2;
    BOOST_REQUIRE_EQUAL(MetaObjectFactory::instance().getObjectSystem()->TestBuildAllRuntimeSourceFiles(cb, true), 0);
    BOOST_REQUIRE_EQUAL(obj1->test, 1);
    BOOST_REQUIRE_EQUAL(obj2->test, 2);
}

#ifdef HAVE_CUDA
BOOST_AUTO_TEST_CASE(test_cuda_recompile)
{
    auto obj = rcc::shared_ptr<test_cuda_object>::create();
    obj->run_kernel();
    BOOST_REQUIRE_EQUAL(MetaObjectFactory::instance().getObjectSystem()->TestBuildAllRuntimeSourceFiles(cb, true), 0);
    obj->run_kernel();
}
#endif

BOOST_AUTO_TEST_CASE(test_object_cleanup)
{
    AUDynArray<IObjectConstructor*> constructors;
    MetaObjectFactory::instance().getObjectSystem()->GetObjectFactorySystem()->GetAll(constructors);
    for (int i = 0; i < constructors.Size(); ++i) {
        BOOST_REQUIRE_EQUAL(constructors[i]->GetNumberConstructedObjects(), 0);
    }
    delete cb;
}
