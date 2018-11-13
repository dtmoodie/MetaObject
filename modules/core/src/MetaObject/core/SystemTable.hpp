#pragma once
#include <MetaObject/core/TypeTable.hpp>
#include <MetaObject/core/detail/Allocator.hpp>
#include <MetaObject/detail/Export.hpp>
#include <MetaObject/detail/TypeInfo.hpp>
#include <MetaObject/logging/logging.hpp>

#include <RuntimeObjectSystem/ObjectInterfacePerModule.h>
#include <RuntimeObjectSystem/shared_ptr.hpp>

#include <map>
#include <memory>

struct MO_EXPORTS ISingletonContainer
{
    virtual ~ISingletonContainer();
};

template <typename T>
struct TSingletonContainer : public ISingletonContainer
{
    T* ptr = nullptr;
};

template <typename T>
struct OwningContainer : public TSingletonContainer<T>
{
    OwningContainer(std::unique_ptr<T>&& ptr_)
        : m_ptr(std::move(ptr_))
    {
        this->ptr = m_ptr.get();
    }

  private:
    std::unique_ptr<T> m_ptr;
};

template <class T, class E>
struct SharingContainer;

template <typename T>
struct SharingContainer<T, typename std::enable_if<std::is_base_of<IObject, T>::value>::type>
    : public TSingletonContainer<T>
{
    SharingContainer(T* ptr_)
        : m_ptr(ptr_)
    {
        this->ptr = m_ptr.get();
    }
    SharingContainer(const rcc::shared_ptr<T>& ptr_)
        : m_ptr(ptr_)
    {
        this->ptr = m_ptr.get();
    }

  private:
    rcc::shared_ptr<T> m_ptr;
};

template <typename T>
struct SharingContainer<T, typename std::enable_if<!std::is_base_of<IObject, T>::value>::type>
    : public TSingletonContainer<T>
{
    SharingContainer(T* ptr_)
        : m_ptr(ptr_)
    {
        this->ptr = m_ptr.get();
    }
    SharingContainer(const std::shared_ptr<T>& ptr_)
        : m_ptr(ptr_)
    {
        this->ptr = m_ptr.get();
    }

  private:
    std::shared_ptr<T> m_ptr;
};

template <typename T>
struct NonOwningContainer : public TSingletonContainer<T>
{
    NonOwningContainer(T* ptr_)
    {
        this->ptr = ptr_;
    }
};

struct SystemInfo
{
    bool have_cuda = false;
};

namespace mo
{
    class MetaObjectFactory;
}

struct MO_EXPORTS SystemTable : std::enable_shared_from_this<SystemTable>
{
  public:
    SystemTable(const SystemTable& other) = delete;
    SystemTable& operator=(const SystemTable& other) = delete;

    static std::shared_ptr<SystemTable> instance();
    static void setInstance(const std::shared_ptr<SystemTable>& table);
    static bool checkInstance();

    SystemTable();
    virtual ~SystemTable();

    template <class T>
    T* getSingleton();

    // Owning
    template <typename T>
    T* setSingleton(std::unique_ptr<T>&& singleton);

    // Shared
    template <typename T>
    T* setSingleton(const rcc::shared_ptr<T>& singleton);

    template <typename T>
    T* setSingleton(const std::shared_ptr<T>& singleton);

    // Non owning
    template <typename T>
    T* setSingleton(T* ptr);

    void deleteSingleton(mo::TypeInfo type);

    template <typename T>
    void deleteSingleton();

    // Members
    mo::MetaObjectFactory* metaobject_factory = nullptr;
    std::shared_ptr<mo::Allocator> allocator;
    SystemInfo system_info;

  private:
    std::map<mo::TypeInfo, std::unique_ptr<ISingletonContainer>> m_singletons;
}; // struct SystemTable

//////////////////////////////////////////////////////////////////////////////
///                              Implementation
//////////////////////////////////////////////////////////////////////////////
template <class T, class U = T>
T* singleton(SystemTable* table)
{
    T* ptr = nullptr;
    if (table)
    {
        ptr = table->getSingleton<T>();
        if (ptr == nullptr)
        {
            ptr = table->setSingleton(std::unique_ptr<T>(new U()));
            MO_LOG(info) << "Creating new " << mo::TypeTable::instance(table).typeToName(mo::TypeInfo(typeid(U)))
                         << " singleton instance " << static_cast<const void*>(ptr)
                         << " in system table: " << static_cast<const void*>(table);
        }
    }
    return ptr;
}

template <class T, class U = T>
T* singleton()
{
    auto module = PerModuleInterface::GetInstance();
    auto table = module->GetSystemTable();
    if (table)
    {
        return singleton<T, U>(table);
    }
    else
    {
        return nullptr;
    }
}

template <class T>
T* SystemTable::getSingleton()
{
    auto itr = m_singletons.find(mo::TypeInfo(typeid(T)));
    if (itr != m_singletons.end())
    {
        auto container = static_cast<TSingletonContainer<T>*>(itr->second.get());
        if (container)
        {
            return container->ptr;
        }
    }
    return nullptr;
}

// Owning
template <typename T>
T* SystemTable::setSingleton(std::unique_ptr<T>&& singleton)
{
    std::unique_ptr<OwningContainer<T>> owner(new OwningContainer<T>(std::move(singleton)));
    T* ptr = owner->ptr;
    m_singletons[mo::TypeInfo(typeid(T))] = std::move(owner);
    return ptr;
}

// Shared
template <typename T>
T* SystemTable::setSingleton(const rcc::shared_ptr<T>& singleton)
{
    std::unique_ptr<SharingContainer<T, void>> owner(new SharingContainer<T, void>(singleton));
    T* ptr = owner->ptr;
    m_singletons[mo::TypeInfo(typeid(T))] = std::move(owner);
    return ptr;
}

template <typename T>
T* SystemTable::setSingleton(const std::shared_ptr<T>& singleton)
{
    m_singletons[mo::TypeInfo(typeid(T))] =
        std::unique_ptr<SharingContainer<T, void>>(new SharingContainer<T, void>(singleton));
    return singleton.get();
}

// Non owning
template <typename T>
T* SystemTable::setSingleton(T* ptr)
{
    std::unique_ptr<NonOwningContainer<T>> owner(new NonOwningContainer<T>(ptr));
    m_singletons[mo::TypeInfo(typeid(T))] = std::move(owner);
    return ptr;
}

template <typename T>
void SystemTable::deleteSingleton()
{
    deleteSingleton(typeid(T));
}
