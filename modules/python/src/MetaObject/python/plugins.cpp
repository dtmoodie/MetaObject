#include "MetaObject/object/MetaObjectFactory.hpp"
#include "PythonSetup.hpp"
#include <RuntimeObjectSystem/RuntimeObjectSystem.h>
#include <boost/optional/optional.hpp>
#include <boost/python.hpp>

namespace mo
{

    bool loadPlugin(std::string str)
    {
        if(mo::MetaObjectFactory::instance()->loadPlugin(str))
        {
            auto module_name = python::getModuleName();
            boost::python::object plugins_module(boost::python::handle<>(
                boost::python::borrowed(PyImport_AddModule((module_name + ".plugins").c_str()))));
            boost::python::import(module_name.c_str()).attr("plugins") = plugins_module;
            // set the current scope to the new sub-module
            boost::python::scope plugins_scope = plugins_module;
            auto plugin_names = mo::MetaObjectFactory::instance()->listLoadedPluginInfo();
            for (auto& name : plugin_names)
            {
                boost::shared_ptr<PluginInfo> plugin(new PluginInfo(name));
                boost::python::import(module_name.c_str()).attr("plugins").attr(name.getPluginName().c_str()) =
                    plugin;
            }
            mo::python::registerInterfaces();
            mo::python::registerObjects();
            return true;
        }
        return false;
    }

    int loadPluginsInternal(std::string dir)
    {
        int nplugins = mo::MetaObjectFactory::instance()->loadPlugins(dir);
        {
            auto module_name = python::getModuleName();
            boost::python::object plugins_module(boost::python::handle<>(
                boost::python::borrowed(PyImport_AddModule((module_name + ".plugins").c_str()))));
            boost::python::import(module_name.c_str()).attr("plugins") = plugins_module;
            // set the current scope to the new sub-module
            boost::python::scope plugins_scope = plugins_module;
            auto plugin_names = mo::MetaObjectFactory::instance()->listLoadedPluginInfo();
            for (auto& name : plugin_names)
            {
                boost::shared_ptr<PluginInfo> plugin(new PluginInfo(name));
                boost::python::import(module_name.c_str()).attr("plugins").attr(name.getPluginName().c_str()) =
                    plugin;
            }
        }
        return nplugins;
    }

    int loadPlugins(std::string dir)
    {
        auto nplugins = loadPluginsInternal(dir);
        mo::python::registerInterfaces();
        mo::python::registerObjects();
        return nplugins;
    }

    std::vector<std::string> getPluginIncludeDirs(const PluginInfo& plugin)
    {
        std::vector<FileSystemUtils::Path>& paths =
            mo::MetaObjectFactory::instance()->getObjectSystem()->GetIncludeDirList(plugin.m_id);
        std::vector<std::string> output;
        for (const auto& path : paths)
            output.push_back(path.m_string);
        return output;
    }

    void addIncludeDir(const PluginInfo& plugin, const std::string& str)
    {
        mo::MetaObjectFactory::instance()->getObjectSystem()->AddIncludeDir(str.c_str(), plugin.m_id);
    }

    std::vector<std::string> getPluginLinkDirs(const PluginInfo& plugin)
    {
        std::vector<FileSystemUtils::Path>& paths =
            mo::MetaObjectFactory::instance()->getObjectSystem()->GetLinkDirList(plugin.m_id);
        std::vector<std::string> output;
        for (const auto& path : paths)
            output.push_back(path.m_string);
        return output;
    }

    void addLinkDir(const PluginInfo& plugin, const std::string& str)
    {
        mo::MetaObjectFactory::instance()->getObjectSystem()->AddLibraryDir(str.c_str(), plugin.m_id);
    }

    std::string getCompileOptions(const PluginInfo& plugin)
    {
        return std::string(
            mo::MetaObjectFactory::instance()->getObjectSystem()->GetAdditionalCompileOptions(plugin.m_id));
    }

    void addCompileOptions(const PluginInfo& plugin, const std::string& str)
    {
        mo::MetaObjectFactory::instance()->getObjectSystem()->AppendAdditionalCompileOptions(str.c_str(), plugin.m_id);
    }

    std::vector<std::string> listLoadedPlugins() { return mo::MetaObjectFactory::instance()->listLoadedPlugins(); }

    void setupPlugins(const std::string& module_name)
    {
        boost::python::object plugins_module(
            boost::python::handle<>(boost::python::borrowed(PyImport_AddModule((module_name + ".plugins").c_str()))));
        boost::python::scope().attr("plugins") = plugins_module;
        // set the current scope to the new sub-module
        boost::python::scope plugins_scope = plugins_module;

        boost::python::class_<PluginInfo, boost::shared_ptr<PluginInfo>, boost::noncopyable>("PluginInfo",
                                                                                             boost::python::no_init)
            .add_property("name", &PluginInfo::getPluginName)
            .add_property("state", &PluginInfo::getState)
            .add_property("build_info", &PluginInfo::getBuildInfo)
            .add_property("id", &PluginInfo::getId)
            .add_property("load_time", &PluginInfo::getLoadTime)
            .def("getIncludeDirs", &getPluginIncludeDirs)
            .def("getLinkDirs", &getPluginLinkDirs)
            .def("getCompileOptions", &getCompileOptions)
            .def("addLinkDir", &addLinkDir)
            .def("addIncludeDir", &addIncludeDir)
            .def("addCompileOptions", &addCompileOptions);

        boost::python::def("loadPlugin", &loadPlugin);
        boost::python::def("loadPlugins", &loadPlugins);
#ifdef _MSC_VER
// loadPluginsInternal("./plugins");
#else
// loadPluginsInternal("./bin/plugins");
#endif
        mo::python::registerInterfaces();
        mo::python::registerObjects();
    }
}
