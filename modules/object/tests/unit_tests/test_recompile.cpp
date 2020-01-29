#include "TestObjects.hpp"

#include "RuntimeObjectSystem/IObjectFactorySystem.h"
#include "RuntimeObjectSystem/RuntimeObjectSystem.h"
#include "RuntimeObjectSystem/shared_ptr.hpp"

#include <MetaObject/core/AsyncStreamFactory.hpp>
#include <MetaObject/object/MetaObjectFactory.hpp>

#include <boost/filesystem.hpp>
#include <iostream>

#include <gtest/gtest.h>

using namespace mo;

namespace
{

    class BuildCallback : public ITestBuildNotifier
    {
        bool TestBuildCallback(const char* file, TestBuildResult type) override;
        bool TestBuildWaitAndUpdate() override;
    };

    bool BuildCallback::TestBuildCallback(const char* file, TestBuildResult type)
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
    bool BuildCallback::TestBuildWaitAndUpdate()
    {
        return true;
    }

    static BuildCallback build_cb{};
} // namespace

TEST(recompile, recompile)
{
    ASSERT_EQ(MetaObjectFactory::instance()->getObjectSystem()->TestBuildAllRuntimeSourceFiles(&build_cb, true), 0);
}

TEST(recompile, object_swap)
{
    auto constructor =
        MetaObjectFactory::instance()->getObjectSystem()->GetObjectFactorySystem()->GetConstructor("MetaObjectSignals");
    auto obj = constructor->Construct();
    ASSERT_NE(obj, nullptr);
    auto control_block = constructor->GetControlBlock(obj->GetPerTypeId());
    ASSERT_NE(control_block, nullptr);
    rcc::shared_ptr<IMetaObject> ptr(control_block);
    rcc::shared_ptr<test::MetaObjectSignals> typed_ptr(ptr);
    test::MetaObjectSignals* old_ptr = typed_ptr;
    ASSERT_TRUE(old_ptr != nullptr);
    ASSERT_EQ(MetaObjectFactory::instance()->getObjectSystem()->TestBuildAllRuntimeSourceFiles(&build_cb, true), 0);
    test::MetaObjectSignals* new_ptr = typed_ptr;
    ASSERT_TRUE(new_ptr != nullptr);
    ASSERT_NE(old_ptr, new_ptr);
    std::vector<SignalInfo*> signals;
    new_ptr->getSignalInfo(signals);
    ASSERT_EQ(signals.size(), 2);
}

TEST(recompile, reinitialize_publisher)
{
    auto publisher = test::MetaObjectSignals::create();

    auto signals = publisher->getSignals();
    std::vector<std::pair<mo::TypeInfo, std::string>> signatures;
    for (const auto& sig : signals)
    {
        signatures.emplace_back(sig.first->getSignature(), sig.second);
    }

    ASSERT_EQ(MetaObjectFactory::instance()->getObjectSystem()->TestBuildAllRuntimeSourceFiles(&build_cb, true), 0);

    signals = publisher->getSignals();
    for (const auto& sig : signatures)
    {
        auto itr = std::find_if(signals.begin(), signals.end(), [&sig](const std::pair<ISignal*, std::string>& pair) {
            return pair.first->getSignature() == sig.first && pair.second == sig.second;
        });
        EXPECT_NE(itr, signals.end()) << "After recompilation the MetaObjectSignals object is missing the signal "
                                      << sig.second << "[ " << sig.first.name() << "]";
    }
}

TEST(recompile, reinitialize_subscriber)
{
    auto publisher = test::MetaObjectSlots::create();

    auto signals = publisher->getSlots();
    std::vector<std::pair<mo::TypeInfo, std::string>> signatures;
    for (const auto& sig : signals)
    {
        signatures.emplace_back(sig.first->getSignature(), sig.second);
    }

    ASSERT_EQ(MetaObjectFactory::instance()->getObjectSystem()->TestBuildAllRuntimeSourceFiles(&build_cb, true), 0);

    signals = publisher->getSlots();
    for (const auto& sig : signatures)
    {
        auto itr = std::find_if(signals.begin(), signals.end(), [&sig](const std::pair<ISlot*, std::string>& pair) {
            return pair.first->getSignature() == sig.first && pair.second == sig.second;
        });
        EXPECT_NE(itr, signals.end()) << "After recompilation the MetaObjectSlots object is missing the slot "
                                      << sig.second << "[ " << sig.first.name() << "]";
    }
}

TEST(recompile, pointer_mechanics)
{
    auto obj = test::MetaObjectSignals::create();
    EXPECT_TRUE(obj);
    auto control_block = obj.GetControlBlock();
    EXPECT_TRUE(control_block);

    EXPECT_EQ(control_block.use_count(), 2);
    // Test construction of weak pointers from shared pointers
    {
        rcc::weak_ptr<test::MetaObjectSignals> weak_ptr(obj);
        EXPECT_EQ(control_block.use_count(), 2);
    }
    EXPECT_EQ(control_block.use_count(), 2);

    {
        rcc::weak_ptr<IMetaObject> weak_ptr_meta_object;
        weak_ptr_meta_object = rcc::weak_ptr<IMetaObject>(obj);
        EXPECT_EQ(control_block.use_count(), 2);
    }
    EXPECT_EQ(control_block.use_count(), 2);

    // Test construction of weak pointers from raw object pointer
    {
        rcc::weak_ptr<IMetaObject> weak_ptr_meta_object(*obj);
        EXPECT_EQ(control_block.use_count(), 2);
    }
    EXPECT_EQ(control_block.use_count(), 2);

    {
        rcc::weak_ptr<IMetaObject> weak_ptr_meta_object = rcc::weak_ptr<IMetaObject>(*obj);
        EXPECT_EQ(control_block.use_count(), 2);
    }

    EXPECT_EQ(control_block.use_count(), 2);

    // Test shared pointer mechanics
    {
        rcc::shared_ptr<test::MetaObjectSignals> shared(obj);
        EXPECT_EQ(control_block.use_count(), 3);
    }
    EXPECT_EQ(control_block.use_count(), 2);

    {
        rcc::shared_ptr<IMetaObject> shared_ptr_meta_object;
        shared_ptr_meta_object = rcc::shared_ptr<IMetaObject>(obj);
        EXPECT_EQ(control_block.use_count(), 3);
    }
    EXPECT_EQ(control_block.use_count(), 2);

    // Test construction of shared pointers from raw object pointer
    {
        rcc::shared_ptr<IMetaObject> shared_ptr_meta_object(*obj);
        EXPECT_EQ(control_block.use_count(), 3);
    }

    EXPECT_EQ(control_block.use_count(), 2);

    {
        rcc::shared_ptr<IMetaObject> shared_ptr_meta_object = rcc::shared_ptr<IMetaObject>(*obj);
        EXPECT_EQ(control_block.use_count(), 3);
    }
    EXPECT_EQ(control_block.use_count(), 2);
}

TEST(object_factory, test_creation_function)
{
    auto obj = test::MetaObjectSignals::create();
    ASSERT_NE(obj, nullptr);
}

TEST(recompile, reconnect_signals)
{
    auto emitter = test::MetaObjectSignals::create();
    auto receiver = test::MetaObjectSlots::create();

    test::MetaObjectSignals* original_emitter = emitter;
    test::MetaObjectSlots* original_receiver = receiver;
    ASSERT_EQ(IMetaObject::connect(*emitter, "test_int", *receiver, "test_int"), 1);
    int value = 5;
    emitter->sig_test_int(value);
    EXPECT_EQ(receiver->slot_called_value, value);
    EXPECT_EQ(receiver->slot_called_count, 1);
    mo::getDefaultLogger().set_level(spdlog::level::trace);
    ASSERT_EQ(MetaObjectFactory::instance()->getObjectSystem()->TestBuildAllRuntimeSourceFiles(&build_cb, true), 0);
    mo::getDefaultLogger().set_level(spdlog::level::info);

    ASSERT_NE(original_receiver, receiver.get());
    ASSERT_NE(original_emitter, emitter.get());

    EXPECT_EQ(receiver->slot_called_count, 0);

    auto signal = emitter->getSignal("test_int", mo::TypeInfo::create<void(int)>());
    ASSERT_NE(signal, nullptr);
    EXPECT_TRUE(signal->isConnected());
    value = 10;
    emitter->sig_test_int(value);
    EXPECT_EQ(receiver->slot_called_value, value);
    EXPECT_EQ(receiver->slot_called_count, 1);

    value = 20;
    emitter->sig_test_int(value);
    EXPECT_EQ(receiver->slot_called_value, value);
    EXPECT_EQ(receiver->slot_called_count, 2);
}

/*TEST(recompile, input_output_param)
{
    auto ctx = mo::AsyncStreamFactory::instance()->create("test_input_output_param");
    auto output = rcc::shared_ptr<test::ParamedObject>::create();
    auto input = rcc::shared_ptr<test_meta_object_input>::create();
    input->setStream(ctx);
    output->setStream(ctx);
    auto output_param = output->getOutput("test_output");
    ASSERT_NE(output_param, nullptr);
    auto input_param = input->getInput("test_input");
    ASSERT_NE(input_param, nullptr);
    ASSERT_EQ(input->param_update_call_count, 0);
    ASSERT_EQ(output->param_update_call_count, 0);
    ASSERT_TRUE(IMetaObject::connectInput(output.get(), output_param, input.get(), input_param));
    ASSERT_EQ(input->param_update_call_count, 1);
    ASSERT_EQ(output->param_update_call_count, 0);
    output->test_output.updateData(5);
    ASSERT_EQ(output->param_update_call_count, 1);
    ASSERT_EQ(input->param_update_call_count, 2);
    ASSERT_NE(input->test_input, nullptr);
    ASSERT_NE(input->test_input_param.getData(), nullptr);
    ASSERT_EQ(*input->test_input, output->test_output.value());
    ASSERT_EQ(MetaObjectFactory::instance()->getObjectSystem()->TestBuildAllRuntimeSourceFiles(&build_cb, true), 0);
    ASSERT_EQ(input->param_update_call_count, 1);
    output->test_output.updateData(10);
    ASSERT_EQ(input->param_update_call_count, 2);
    ASSERT_NE(input->test_input_param.getData(), nullptr);
    ASSERT_NE(input->test_input, nullptr);
    ASSERT_EQ(*input->test_input, output->test_output.value());
    ASSERT_EQ(*input->test_input, 10);
}*/

/*TEST(recompile, param_persistence_recompile)
{
    auto obj = test_meta_object_parameters::create();
    ASSERT_EQ(obj->test, 5);
    obj->test = 10;
    ASSERT_EQ(MetaObjectFactory::instance()->getObjectSystem()->TestBuildAllRuntimeSourceFiles(&build_cb, true), 0);
    ASSERT_EQ(obj->test, 10);
}

TEST(recompile, multiple_objects)
{
    auto obj1 = rcc::shared_ptr<test_meta_object_parameters>::create();
    auto obj2 = rcc::shared_ptr<test_meta_object_parameters>::create();
    obj1->test = 1;
    obj2->test = 2;
    ASSERT_EQ(MetaObjectFactory::instance()->getObjectSystem()->TestBuildAllRuntimeSourceFiles(&build_cb, true), 0);
    ASSERT_EQ(obj1->test, 1);
    ASSERT_EQ(obj2->test, 2);
}*/

TEST(recompile, object_cleanup)
{
    AUDynArray<IObjectConstructor*> constructors;
    MetaObjectFactory::instance()->getObjectSystem()->GetObjectFactorySystem()->GetAll(constructors);
    for (size_t i = 0; i < constructors.Size(); ++i)
    {
        auto num_objects = constructors[i]->GetNumberConstructedObjects();
        for (size_t j = 0; j < num_objects; ++j)
        {
            auto obj = constructors[i]->GetConstructedObject(j);
            EXPECT_EQ(obj, nullptr) << " object not cleaned up for " << constructors[i]->GetName();
        }
    }
}
