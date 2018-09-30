#pragma once
#include "MetaObject/python/DataConverter.hpp"
#include <MetaObject/params/IParam.hpp>

#include <boost/python.hpp>

#include <memory>

namespace mo
{
    namespace python
    {

        void setupParameters(const std::string& module_name);

        struct ParamCallbackContainer
        {
            using Ptr = std::unique_ptr<ParamCallbackContainer>;
            using Registry = std::map<mo::ParamBase*, std::vector<Ptr>>;

            static std::shared_ptr<Registry> registry();
            ParamCallbackContainer(mo::IParam* ptr, const boost::python::object& obj);

            void onParamUpdate(IParam* param,
                               Context*,
                               OptionalTime_t ts,
                               size_t fn,
                               const std::shared_ptr<ICoordinateSystem>&,
                               UpdateFlags flags);

            void onParamDelete(const mo::ParamBase*);

          private:
            mo::IParam* m_ptr = nullptr;
            boost::python::object m_callback;
            mo::python::DataConverterRegistry::Get_t m_getter;
            UpdateSlot_t m_slot;
            DeleteSlot_t m_delete_slot;
            std::shared_ptr<Connection> del_connection;
            std::shared_ptr<Connection> update_connection;
        };
    }
}
