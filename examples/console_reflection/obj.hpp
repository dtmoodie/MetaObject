#pragma once
#include <MetaObject/object/MetaObject.hpp>
#include <MetaObject/object/MetaObjectInfo.hpp>
#include <MetaObject/object/detail/MetaObjectMacros.hpp>

/*!
 * \brief The ExampleInterfaceInfo struct for static information about objects that inherit from the ExampleInterface
 * class
 */
struct ExampleInterfaceInfo : public mo::IMetaObjectInfo
{
    virtual void PrintHelp() const = 0;
};

namespace mo
{
    /*!
     *  This template specialization deals with the concrete object 'Type' which in this example
     *  must have a static function called PrintHelp.
     */
    template <class Type>
    struct MetaObjectInfoImpl<Type, ExampleInterfaceInfo> : public ExampleInterfaceInfo
    {
        /*!
         * \brief PrintHelp calls the static function PrintHelp in the concrete implementation
         *        'Type'
         */
        virtual void PrintHelp() const
        {
            return Type::PrintHelp();
        }
    };
} // namespace mo

/*!
 * \brief The ExampleInterface class contains one virtual member foo and the typedef InterfaceInfo
 *
 */
class ExampleInterface : public TInterface<ExampleInterface, mo::MetaObject>
{
  public:
    /*!
     * \brief InterfaceInfo typedef allows for the MetaObjectInfo templated class in
     * MetaObject/object/MetaObjectInfo.hpp
     *        to detect the correct object info interface to inherit from
     */
    typedef ExampleInterfaceInfo InterfaceInfo;

    // These macros are needed to initialize some reflection code

    MO_BEGIN(ExampleInterface)
    INTERNAL_MEMBER_FUNCTION(foo)
    MO_END;

    // The one virtual function to be called from this interface.
    virtual void foo() = 0;
};
