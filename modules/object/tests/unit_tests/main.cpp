#include "TestObjects.hpp"
#include <MetaObject/core/SystemTable.hpp>
#include <MetaObject/logging/logging.hpp>
#include <MetaObject/object/MetaObjectFactory.hpp>
#include <MetaObject/params.hpp>
#include <MetaObject/thread/FiberScheduler.hpp>
#include <MetaObject/thread/ThreadPool.hpp>

#include <boost/filesystem.hpp>

#include <gtest/gtest.h>

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    auto table = SystemTable::instance();
    PerModuleInterface::GetInstance()->SetSystemTable(table.get());
    mo::params::init(table.get());
    mo::MetaObjectFactory::instance()->registerTranslationUnit();
    test::setupPlugin(table.get());

    MO_LOG(info, "Current working directory {}", boost::filesystem::current_path().string());
    std::string postfix;
#ifdef _DEBUG
    postfix = "d";
#endif
    table->getDefaultLogger()->set_level(spdlog::level::debug);
#ifdef _MSC_VER
    const bool loaded = mo::MetaObjectFactory::instance()->loadPlugin("./mo_objectplugin" + postfix + ".dll");
#else
    const bool loaded = mo::MetaObjectFactory::instance()->loadPlugin("./libmo_objectplugin" + postfix + ".so");
#endif

    table->getDefaultLogger()->set_level(spdlog::level::info);
    if (!loaded)
    {
        MO_LOG(warn,
               "Unable to load objectplugin shared library, most tests wont work, this is okay if you're just querying "
               "for available tests");
    }

    auto result = RUN_ALL_TESTS();
    return result;
}
