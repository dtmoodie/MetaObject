#include "MetaObject/object/MetaObjectFactory.hpp"
#include "PythonSetup.hpp"
#include <boost/optional/optional.hpp>
#include <boost/python.hpp>

namespace mo
{
    bool loadPlugin(const std::string& str) { return mo::MetaObjectFactory::instance()->loadPlugin(str); }

    int loadPlugins(const std::string& dir)
    {

        int nplugins = mo::MetaObjectFactory::instance()->loadPlugins(dir);
        {
            boost::python::object plugins_module(
                boost::python::handle<>(boost::python::borrowed(PyImport_AddModule("metaobject.plugins"))));
            boost::python::import("metaobject").attr("plugins") = plugins_module;
            // set the current scope to the new sub-module
            boost::python::scope plugins_scope = plugins_module;
            auto plugin_names = mo::MetaObjectFactory::instance()->listLoadedPluginInfo();
            for (auto& name : plugin_names)
            {
                boost::shared_ptr<PluginInfo> plugin(new PluginInfo(name));
                boost::python::import("metaobject").attr("plugins").attr(name.getPluginName().c_str()) = plugin;
            }
        }

        mo::python::registerObjects();
        return nplugins;
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
            .add_property("load_time", &PluginInfo::getLoadTime);

        boost::python::def("loadPlugin", &loadPlugin);
        boost::python::def("loadPlugins", &loadPlugins);
        loadPlugins("./bin/Plugins");
    }
}
