#pragma once


#include <MetaObject/MetaObject.hpp>

/*!
 * \brief The ExampleInterfaceInfo struct for static information about objects that inherit from the ExampleInterface class
 */
struct ExampleInterfaceInfo: public mo::IMetaObjectInfo
{
    virtual void PrintHelp() = 0;
};


namespace mo
{
    /*!
     *  This template specialization deals with the concrete object 'Type' which in this example
     *  must have a static function called PrintHelp.
     */
    template<class Type>
    struct MetaObjectInfoImpl<Type, ExampleInterfaceInfo>: public ExampleInterfaceInfo
    {
        /*!
         * \brief PrintHelp calls the static function PrintHelp in the concrete implementation
         *        'Type'
         */
        virtual void PrintHelp()
        {
            return Type::PrintHelp();
        }
    };
}

/*!
 * \brief The ExampleInterface class contains one virtual member foo and the typedef InterfaceInfo
 *
 */
class ExampleInterface: public TInterface<ctcrc32("ExampleInterface"), mo::IMetaObject>
{
public:
    MO_BEGIN(ExampleInterface)
    MO_END
    typedef ExampleInterfaceInfo InterfaceInfo;
    virtual void foo() = 0;
};
