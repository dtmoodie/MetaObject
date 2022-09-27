#include "TestObjects.hpp"
#include <MetaObject/core/SystemTable.hpp>
#include <MetaObject/logging/logging.hpp>
#include <MetaObject/object/MetaObjectFactory.hpp>
#include <MetaObject/params.hpp>
#include <MetaObject/thread/FiberScheduler.hpp>
#include <MetaObject/thread/ThreadPool.hpp>

#include "mo_objectplugin_export.hpp"

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
        table->getLogger()->set_level(spdlog::level::critical);
    }
    PerModuleInterface::GetInstance()->SetSystemTable(table.get());
    mo::params::init(table.get());
    std::shared_ptr<mo::MetaObjectFactory> factory = mo::MetaObjectFactory::instance(table.get());
    mo_objectplugin::initPlugin(0, factory.get());

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
        table->getLogger()->set_level(spdlog::level::debug);
    }

    mo::MetaObjectFactory::loadStandardPlugins();

    if (!quiet)
    {
        table->getLogger()->set_level(spdlog::level::info);
    }

    auto result = RUN_ALL_TESTS();
    return result;
}
