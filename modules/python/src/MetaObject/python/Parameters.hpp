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
            using Ptr_t = std::unique_ptr<ParamCallbackContainer>;
            using Registry_t = std::map<IParam*, std::vector<Ptr_t>>;

            static std::shared_ptr<Registry_t> registry();
            ParamCallbackContainer(mo::IControlParam* ptr, const boost::python::object& obj);

            void onParamUpdate(const IParam&, Header, UpdateFlags, IAsyncStream&);

            void onParamDelete(const IParam&);

          private:
            mo::IControlParam* m_ptr = nullptr;
            boost::python::object m_callback;
            // mo::python::DataConverterRegistry::Get_t m_getter;
            UpdateSlot_t m_slot;
            DeleteSlot_t m_delete_slot;
            std::shared_ptr<Connection> del_connection;
            std::shared_ptr<Connection> update_connection;
        };
    } // namespace python
} // namespace mo
