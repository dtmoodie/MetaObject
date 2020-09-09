#ifndef MO_PYTHON_DATACONVERTER_HPP
#define MO_PYTHON_DATACONVERTER_HPP
#include "converters.hpp"

#include <MetaObject/core/TypeTable.hpp>
#include <MetaObject/detail/Export.hpp>
#include <MetaObject/params/AccessToken.hpp>
#include <MetaObject/params/IControlParam.hpp>
#include <MetaObject/params/IPublisher.hpp>
#include <MetaObject/params/ITControlParam.hpp>
#include <MetaObject/python/PythonSetup.hpp>

#include <boost/python.hpp>
#include <boost/python/extract.hpp>

#include <ct/interop/boost_python/PythonConverter.hpp>

#include <map>

namespace mo
{
    namespace python
    {
        struct MO_EXPORTS DataConversionTable
        {
            using FromPython_t = std::function<bool(void*, ITraits* trait, const boost::python::object&)>;
            using ToPython_t = std::function<boost::python::object(const void*, ITraits* trait)>;

            static DataConversionTable* instance();
            static DataConversionTable* instance(SystemTable* table);

            void registerConverters(const mo::TypeInfo& type, const FromPython_t& from, const ToPython_t& to);

            template <class T>
            void registerConverters(const FromPython_t& from, const ToPython_t& to)
            {
                this->registerConverters(mo::TypeInfo::create<T>(), from, to);
            }

            ToPython_t getConverterToPython(const mo::TypeInfo& type) const;
            FromPython_t getConverterFromPython(const mo::TypeInfo& type) const;
            std::vector<mo::TypeInfo> listConverters() const;

          private:
            std::map<mo::TypeInfo, std::pair<ToPython_t, FromPython_t>> m_registered_converters;
        };

    } // namespace python
} // namespace mo

#endif // MO_PYTHON_DATACONVERTER_HPP