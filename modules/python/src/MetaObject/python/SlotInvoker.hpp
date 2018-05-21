#pragma once
#include <boost/python.hpp>
#include <MetaObject/signals/TSlot.hpp>
#include <RuntimeObjectSystem/shared_ptr.hpp>

namespace mo
{
namespace python
{

template<class T, class Sig>
struct SlotInvoker;

template<class T, class R>
struct SlotInvoker<T, R(void)>
{
    static R invoke(const std::string& slot_name, T& obj)
    {
        ISlot* slot = obj.getSlot(slot_name, mo::TypeInfo(typeid(R())));
        if(slot)
        {
            auto tslot = dynamic_cast<mo::TSlot<R()>*>(slot);
            if(tslot)
            {
                return (*tslot)();
            }
        }
    }
};


template<class T, class R, class ... Args>
struct SlotInvoker<T, R(Args...)>
{
    static R invoke(const std::string& slot_name, T& obj, const Args&... args)
    {
        ISlot* slot = obj.getSlot(slot_name, mo::TypeInfo(typeid(R(Args...))));
        if(slot)
        {
            auto tslot = dynamic_cast<mo::TSlot<R(Args...)>*>(slot);
            if(tslot)
            {
                return (*tslot)(args...);
            }
        }
    }
};


} // namespace mo::python
} // namespace mo
