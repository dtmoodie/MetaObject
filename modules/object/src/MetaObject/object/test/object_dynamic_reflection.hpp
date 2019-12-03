#pragma once
#include "../IMetaObjectInfo.hpp"
#include <MetaObject/params/ParamInfo.hpp>
#include <MetaObject/params/TInputParam.hpp>
#include <MetaObject/params/TParamOutput.hpp>
#include <MetaObject/signals/TSignal.hpp>
#include <MetaObject/signals/TSlot.hpp>

#include <ct/reflect.hpp>

#include "gtest/gtest.h"

namespace mo
{
    template <class T, class U, class SIG, ct::Flag_t FLAGS, class MDATA>
    void testDynamicReflection(T& inst, ct::MemberObjectPointer<mo::TSignal<SIG> U::*, FLAGS, MDATA> ptr)
    {
        auto name = ptr.getName().toString();
        auto sig = inst.getSignal(name, mo::TypeInfo::create<SIG>());
        ASSERT_EQ(&ptr.get(inst), sig);
        ASSERT_EQ(sig->getSignature(), mo::TypeInfo::create<SIG>());
    }

    template <class T, class U, class SIG, ct::Flag_t FLAGS, class MDATA>
    void testDynamicReflection(T& inst, ct::MemberObjectPointer<mo::TSlot<SIG> U::*, FLAGS, MDATA> ptr)
    {
        auto name = ptr.getName().toString();
        auto slot = inst.getSlot(name, mo::TypeInfo::create<SIG>());
        ASSERT_EQ(&ptr.get(inst), slot);
        ASSERT_EQ(slot->getSignature(), mo::TypeInfo::create<SIG>());
    }

    template <class T, class U, class D, ct::Flag_t FLAGS, class MDATA>
    void testDynamicReflection(T& inst, ct::MemberObjectPointer<mo::TParamPtr<D> U::*, FLAGS, MDATA> ptr)
    {
        auto name = ptr.getName().slice(0, -6).toString();
        auto param = inst.getParam(name);
        ASSERT_EQ(&ptr.get(inst), param);
        ASSERT_EQ(param->getTypeInfo(), mo::TypeInfo::create<D>());

        auto tdata = param->template getTypedData<D>();
        ASSERT_NE(tdata, nullptr);
    }

    template <class T, class U, class D, ct::Flag_t FLAGS, class MDATA>
    void testDynamicReflection(T& inst, ct::MemberObjectPointer<D U::*, FLAGS, MDATA> ptr)
    {
        auto name = ptr.getName().toString();
        auto initializer = ptr.getMetadata();
        ASSERT_EQ(ptr.get(inst), initializer.getInitialValue());
    }

    template <class T, class U, class D, ct::Flag_t FLAGS, class MDATA>
    void testDynamicReflection(T& inst, ct::MemberObjectPointer<mo::TInputParamPtr<D> U::*, FLAGS, MDATA> ptr)
    {
        auto name = ptr.getName().slice(0, -6).toString();
        auto input = inst.getInput(name);
        ASSERT_EQ(input, &ptr.get(inst));
    }

    template <class T, class U, class D, uint64_t PARAM_FLAGS, ct::Flag_t FLAGS, class MDATA>
    void testDynamicReflection(T& inst,
                               ct::MemberObjectPointer<mo::TParamOutput<D, PARAM_FLAGS> U::*, FLAGS, MDATA> ptr)
    {
        auto name = ptr.getName().toString();
        auto output = inst.getOutput(name);
        if (!output)
        {
            std::cout << "Unable to retrieve output '" << name << "' with flags: " << ptr.get(inst).getFlags()
                      << std::endl;
        }
        ASSERT_EQ(output, &ptr.get(inst));
    }

    template <class T, class BASE, ct::Flag_t FLAGS, class MDATA, class... PTRS>
    void testDynamicReflection(T&, ct::MemberFunctionPointers<BASE, FLAGS, MDATA, PTRS...>)
    {
    }

    template <class T>
    void testDynamicReflectionRecurse(T& inst, ct::Indexer<0> idx)
    {
        auto ptr = ct::Reflect<T>::getPtr(idx);
        testDynamicReflection(inst, ptr);
    }

    template <class T, ct::index_t I>
    void testDynamicReflectionRecurse(T& inst, ct::Indexer<I> idx)
    {
        auto ptr = ct::Reflect<T>::getPtr(idx);
        testDynamicReflection(inst, ptr);
        testDynamicReflectionRecurse(inst, --idx);
    }

    template <class T>
    void testDynamicReflection(rcc::shared_ptr<T>& inst)
    {
        ASSERT_NE(inst, nullptr);
        testDynamicReflectionRecurse(*inst, ct::Reflect<T>::end());
    }

    // static reflection

    template <class T, class U, class SIG, ct::Flag_t FLAGS, class MDATA>
    void testStaticReflection(const IMetaObjectInfo& info, ct::MemberObjectPointer<mo::TSignal<SIG> U::*, FLAGS, MDATA> ptr)
    {
    }

    template <class T, class U, class SIG, ct::Flag_t FLAGS, class MDATA>
    void testStaticReflection(const IMetaObjectInfo& info, ct::MemberObjectPointer<mo::TSlot<SIG> U::*, FLAGS, MDATA> ptr)
    {
    }

    template <class T, class U, class D, ct::Flag_t FLAGS, class MDATA>
    void testStaticReflection(const IMetaObjectInfo& info, ct::MemberObjectPointer<mo::TParamPtr<D> U::*, FLAGS, MDATA> ptr)
    {
    }

    template <class T, class U, class D, ct::Flag_t FLAGS, class MDATA>
    void testStaticReflection(const IMetaObjectInfo& info, ct::MemberObjectPointer<D U::*, FLAGS, MDATA> ptr)
    {
        auto name = ptr.getName();
        auto params = info.getParamInfo();
        for (auto param : params)
        {
            if (name == param->getName())
            {
                if (param->getParamFlags().test(ParamFlags::kINPUT))
                {
                    EXPECT_EQ(
                        param->getDataType(),
                        mo::TypeInfo::create<typename std::remove_pointer<typename std::remove_const<D>::type>::type>())
                        << "Parameter types not equal for " << name << " which is a " << param->getParamFlags();
                }
                else
                {
                    EXPECT_EQ(param->getDataType(), mo::TypeInfo::create<D>())
                        << "Parameter types not equal for " << name << " which is a " << param->getParamFlags();
                }

                return;
            }
        }
        EXPECT_TRUE(false);
    }

    template <class T, class U, class D, ct::Flag_t FLAGS, class MDATA>
    void testStaticReflection(const IMetaObjectInfo& info,
                              ct::MemberObjectPointer<mo::TInputParamPtr<D> U::*, FLAGS, MDATA> ptr)
    {
    }

    template <class T, class U, class D, uint64_t PARAM_FLAGS, ct::Flag_t FLAGS, class MDATA>
    void testStaticReflection(const IMetaObjectInfo& info,
                              ct::MemberObjectPointer<mo::TParamOutput<D, PARAM_FLAGS> U::*, FLAGS, MDATA> ptr)
    {
    }

    template <class T, class BASE, ct::Flag_t FLAGS, class MDATA, class... PTRS>
    void testStaticReflection(const IMetaObjectInfo&, ct::MemberFunctionPointers<BASE, FLAGS, MDATA, PTRS...>)
    {
    }

    template <class T>
    void testStaticReflectionRecurse(const IMetaObjectInfo& info, ct::Indexer<0> idx)
    {
        auto ptr = ct::Reflect<T>::getPtr(idx);
        testStaticReflection<T>(info, ptr);
    }

    template <class T, ct::index_t I>
    void testStaticReflectionRecurse(const IMetaObjectInfo& info, ct::Indexer<I> idx)
    {
        auto ptr = ct::Reflect<T>::getPtr(idx);
        testStaticReflection<T>(info, ptr);
        testStaticReflectionRecurse<T>(info, --idx);
    }

    template <class T>
    void testStaticReflection(const IObjectInfo* info)
    {
        auto obj_info = dynamic_cast<const IMetaObjectInfo*>(info);
        ASSERT_NE(obj_info, nullptr);
        testStaticReflectionRecurse<T>(*obj_info, ct::Reflect<T>::end());
    }
}
