#ifndef MO_PARAMS_ITCONTROLPARAM_HPP
#define MO_PARAMS_ITCONTROLPARAM_HPP
#include <MetaObject/params/IControlParam.hpp>
#include <MetaObject/params/TParam.hpp>
namespace mo
{
    template <class T>
    struct ITControlParam : TParam<IControlParam>
    {
        ITControlParam()
        {
            this->setFlags(ParamFlags::kCONTROL);
        }
        // THESE ARE NOT THREADSAFE.  Since it is expected that you may be trying to mutate a parameter
        // in place, these put the onus on the user to lock for the duration of access
        TypeInfo getTypeInfo() const override
        {
            return TypeInfo::create<T>();
        }
        virtual T& getValue() = 0;
        virtual const T& getValue() const = 0;
        virtual void setValue(T&& val) = 0;
    };
} // namespace mo

#endif // MO_PARAMS_ITCONTROLPARAM_HPP