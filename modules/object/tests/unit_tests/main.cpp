#include "TestObjects.hpp"
#include <MetaObject/core/SystemTable.hpp>
#include <MetaObject/logging/logging.hpp>
#include <MetaObject/object/MetaObjectFactory.hpp>
#include <MetaObject/params.hpp>
#include <MetaObject/thread/FiberScheduler.hpp>
#include <MetaObject/thread/ThreadPool.hpp>

#include <boost/filesystem.hpp>

#include <gtest/gtest.h>

bool beQuiet(int argc, char** argv)
{
    for (int i = 0; i < argc; ++i)
    {
        if (std::string("--gtest_list_tests") == argv[i])
        {
            return true;
        }
    }
    return false;
}

int main(int argc, char** argv)
{
    const auto quiet = beQuiet(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    auto table = SystemTable::instance();
    if (quiet)
    {
        table->getDefaultLogger()->set_level(spdlog::level::critical);
    }
    PerModuleInterface::GetInstance()->SetSystemTable(table.get());
    mo::params::init(table.get());
    mo::MetaObjectFactory::instance()->registerTranslationUnit();
    test::setupPlugin(table.get());
    if (!quiet)
    {
        MO_LOG(info, "Current working directory {}", boost::filesystem::current_path().string());
    }

    std::string postfix;
#ifdef _DEBUG
    postfix = "d";
#endif
    if (!quiet)
    {
        table->getDefaultLogger()->set_level(spdlog::level::debug);
    }

    mo::MetaObjectFactory::loadStandardPlugins();
#ifdef _MSC_VER
    const bool loaded =
        mo::MetaObjectFactory::instance()->loadPlugin("./bin/plugins/mo_objectplugin" + postfix + ".dll");
#else
    const bool loaded =
        mo::MetaObjectFactory::instance()->loadPlugin("./bin/plugins/libmo_objectplugin" + postfix + ".so");
#endif
    if (!quiet)
    {
        table->getDefaultLogger()->set_level(spdlog::level::info);
    }
    if (!loaded)
    {
        MO_LOG(warn,
               "Unable to load objectplugin shared library, most tests wont work, this is okay if you're just querying "
               "for available tests");
    }

    auto result = RUN_ALL_TESTS();
    return result;
}
