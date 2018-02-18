#pragma once
#include <MetaObject/core/detail/Allocator.hpp>
#include <MetaObject/detail/Export.hpp>
#include <MetaObject/detail/TypeInfo.hpp>
#include <RuntimeObjectSystem/shared_ptr.hpp>

#include <map>
#include <memory>

struct MO_EXPORTS ISingletonContainer
{
    virtual ~ISingletonContainer() {}
};

template <typename T>
struct TSingletonContainer : public ISingletonContainer
{
    TSingletonContainer(const std::shared_ptr<T>& ptr_) : ptr(ptr_) {}

    std::shared_ptr<T> ptr;
};

template <typename T>
struct TIObjectSingletonContainer : public ISingletonContainer
{
    TIObjectSingletonContainer(T* ptr_) : ptr(ptr_) {}
    TIObjectSingletonContainer(const rcc::shared_ptr<T>& ptr_) : ptr(ptr_) {}

    rcc::shared_ptr<T> ptr;
};

template <class Derived>
struct TDerivedSystemTable;

struct MO_EXPORTS SystemTable : std::enable_shared_from_this<SystemTable>
{
  public:
    static std::shared_ptr<SystemTable> instance();
    static bool checkInstance();

    SystemTable();
    virtual ~SystemTable();

    std::shared_ptr<mo::Allocator> allocator;

    template <typename T>
    std::enable_if_t<!std::is_base_of<IObject, T>::value, std::shared_ptr<T>> getSingleton()
    {
        auto g_itr = g_singletons.find(mo::TypeInfo(typeid(T)));
        if (g_itr != g_singletons.end())
        {
            return static_cast<TSingletonContainer<T>*>(g_itr->second.get())->ptr;
        }
        return nullptr;
    }

    template <typename T>
    std::enable_if_t<std::is_base_of<IObject, T>::value, rcc::shared_ptr<T>> getSingleton()
    {
        auto g_itr = g_singletons.find(mo::TypeInfo(typeid(T)));
        if (g_itr != g_singletons.end())
        {
            return static_cast<TIObjectSingletonContainer<T>*>(g_itr->second.get())->ptr;
        }
        return {};
    }

    template <typename T>
    typename std::enable_if<!std::is_base_of<IObject, T>::value, T>::type*
    setSingleton(const std::shared_ptr<T>& singleton)
    {
        g_singletons[mo::TypeInfo(typeid(T))] = std::make_shared<TSingletonContainer<T>>(singleton);
        return singleton.get();
    }

    template <typename T>
    typename std::enable_if<std::is_base_of<IObject, T>::value, T>::type*
    setSingleton(const rcc::shared_ptr<T>& singleton)
    {
        g_singletons[mo::TypeInfo(typeid(T))] = std::shared_ptr<TIObjectSingletonContainer<T>>(singleton);
        return singleton.get();
    }

    void deleteSingleton(mo::TypeInfo type);

    template <typename T>
    void deleteSingleton()
    {
        deleteSingleton(mo::TypeInfo(typeid(T)));
    }

  protected:
    static std::weak_ptr<SystemTable> inst;

    template <class Derived>
    friend struct TDerivedSystemTable;

  private:
    std::map<mo::TypeInfo, std::shared_ptr<ISingletonContainer>> g_singletons;
};

template <class Derived>
struct TDerivedSystemTable : public SystemTable
{
    static std::shared_ptr<Derived> instance()
    {
        auto current = SystemTable::instance();
        std::shared_ptr<Derived> output = std::dynamic_pointer_cast<Derived>(current);
        if (!output)
        {
            output = std::make_shared<Derived>();
            SystemTable::inst = output;
        }
        return output;
    }

  protected:
    TDerivedSystemTable() {}
    ~TDerivedSystemTable() {}
};
