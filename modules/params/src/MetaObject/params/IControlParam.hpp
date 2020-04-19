#ifndef MO_PARAMS_ICONTROLPARAM_HPP
#define MO_PARAMS_ICONTROLPARAM_HPP
#include <MetaObject/params/IParam.hpp>
namespace mo
{
    struct MO_EXPORTS IControlParam : IParam
    {
        IControlParam();

        virtual TypeInfo getTypeInfo() const = 0;

        // Check if has been modified
        virtual bool getModified() const = 0;

        // Set if it has been modified
        virtual void setModified(bool value) = 0;
    };
} // namespace mo
#endif // MO_PARAMS_ICONTROLPARAM_HPP