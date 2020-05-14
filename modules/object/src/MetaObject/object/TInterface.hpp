#ifndef MO_OBJECT_TINTERFACE_HPP
#define MO_OBJECT_TINTERFACE_HPP

#include <RuntimeObjectSystem/IObject.h>
#include <RuntimeObjectSystem/InterfaceDatabase.hpp>

#include <ct/StringView.hpp>
#include <ct/VariadicTypedef.hpp>
#include <ct/config.hpp>
#include <ct/hash.hpp>

template <class TInterface>
struct RegisterInterface
{
    RegisterInterface()
    {
        rcc::InterfaceDatabase::RegisterInterface(TInterface::GetInterfaceName(),
                                                  TInterface::getHash(),
                                                  &TInterface::InheritsFrom,
                                                  &TInterface::DirectlyInheritsFrom);
    }
};

template <class T, class... U>
struct TObjectControlBlockImpl<T, ct::VariadicTypedef<U...>, void> : virtual TObjectControlBlock<U>...
{
    TObjectControlBlockImpl(T* obj)
        : TObjectControlBlock<U>(obj)...
    {
        m_obj = obj;
        setObject(obj, ct::VariadicTypedef<U...>());
    }

    void GetTypedObject(T** ret) const
    {
        *ret = m_obj;
    }

    void SetObject(IObject* obj) override
    {
        setObject(obj, ct::VariadicTypedef<U...>());
        m_obj = dynamic_cast<T*>(obj);
    }

    void SetTypedObject(T* obj)
    {
        m_obj = obj;
    }

  private:
    T* m_obj = nullptr;

    template <class U1>
    void setObject(IObject* obj, ct::VariadicTypedef<U1>)
    {
        TObjectControlBlock<U1>::SetObject(obj);
    }

    template <class U1, class... Us>
    void setObject(IObject* obj, ct::VariadicTypedef<U1, Us...>)
    {
        TObjectControlBlock<U1>::SetObject(obj);
        setObject(obj, ct::VariadicTypedef<Us...>());
    }
};

// Template to help with IIDs
template <typename TInferior, typename TSuper, size_t Version = 0>
struct TInterface : public TSuper
{
    using BaseTypes = ct::VariadicTypedef<TSuper>;

    TInterface()
    {
        (void)&s_register_interface;
    }

    static uint32_t getHash()
    {
        return ct::crc32(CT_FUNCTION_NAME);
    }

    static const InterfaceID s_interfaceID;

    static size_t GetInterfaceVersion()
    {
        return Version;
    }

    static size_t GetInterfaceAbiHash()
    {
        size_t seed = Version;
        seed ^= TSuper::GetInterfaceAbiHash() + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }

    static std::string GetInterfaceName()
    {
#ifdef _MSC_VER
        return std::string(__FUNCTION__)
            .substr(ct::findFirst(__FUNCTION__, ' ') + 1,
                    ct::findFirst(__FUNCTION__, ',') - ct::findFirst(__FUNCTION__, ' ') - 1);
#else
        std::string str = __PRETTY_FUNCTION__;
        auto pos1 = str.find("TInferior = ");
        return str.substr(pos1 + 12, str.find(';', pos1 + 13) - pos1 - 12);
#endif
    }

    static bool InheritsFrom(InterfaceID iid)
    {
        if (iid == TInterface::getHash())
        {
            return true;
        }
        return TSuper::InheritsFrom(iid);
    }

    static bool DirectlyInheritsFrom(InterfaceID iid)
    {
        return iid == TSuper::getHash();
    }

    void* GetInterface(InterfaceID _iid) override
    {
        if (_iid == getHash())
        {
            return this;
        }
        return TSuper::GetInterface(_iid);
    }

  private:
    static RegisterInterface<TInterface<TInferior, TSuper>> s_register_interface;
};

template <typename TInferior, typename TSuper, size_t Version>
const InterfaceID
    TInterface<TInferior, TSuper, Version>::s_interfaceID = TInterface<TInferior, TSuper, Version>::getHash();

template <typename TInferior, typename TSuper, size_t Version>
RegisterInterface<TInterface<TInferior, TSuper>> TInterface<TInferior, TSuper, Version>::s_register_interface;

#endif // MO_OBJECT_TINTERFACE_HPP
