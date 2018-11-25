#pragma once
#include <MetaObject/core/TypeTable.hpp>
#include <MetaObject/core/detail/Allocator.hpp>
#include <MetaObject/core/detail/Forward.hpp>
#include <MetaObject/detail/Export.hpp>
#include <MetaObject/detail/TypeInfo.hpp>
#include <MetaObject/detail/defines.hpp>
#include <MetaObject/logging/logging.hpp>

#include <RuntimeObjectSystem/ObjectInterfacePerModule.h>
#include <RuntimeObjectSystem/shared_ptr.hpp>

#include <memory>
#include <unordered_map>

struct SystemInfo
{
    bool have_cuda = false;
};

namespace mo
{
    class MetaObjectFactory;
    struct ISingletonContainer;
}

struct MO_EXPORTS SystemTable : std::enable_shared_from_this<SystemTable>
{
  public:
    using Ptr_t = std::shared_ptr<SystemTable>;

    SystemTable(const SystemTable& other) = delete;
    SystemTable(SystemTable&& other) = delete;
    SystemTable& operator=(const SystemTable& other) = delete;
    SystemTable& operator=(SystemTable&& other) = delete;

    static std::shared_ptr<SystemTable> instance();

    virtual ~SystemTable();

    template <class T>
    T* getSingleton();

    template <class T>
    std::shared_ptr<T> getSharedSingleton();

    // Owning
    template <typename T>
    T* setSingleton(std::unique_ptr<T>&& singleton);

    // Shared
    template <typename T>
    T* setSingleton(const rcc::shared_ptr<T>& singleton);

    template <typename T>
    std::shared_ptr<T> setSingleton(const std::shared_ptr<T>& singleton);

    // Non owning
    template <typename T>
    T* setSingleton(T* ptr);

    void deleteSingleton(mo::TypeInfo type);

    template <typename T>
    void deleteSingleton();

    // Members
    void setAllocatorConstructor(std::function<mo::AllocatorPtr_t()>&& ctr);
    mo::AllocatorPtr_t createAllocator() const;

    mo::AllocatorPtr_t getDefaultAllocator();

    static MO_INLINE void staticDispatchToSystemTable(std::function<void(SystemTable*)>&& func);

  protected:
    static void setInstance(const Ptr_t& table);

  private:
    SystemTable();
    mo::MetaObjectFactory* m_metaobject_factory = nullptr;
    mo::AllocatorPtr_t m_default_allocator;
    SystemInfo m_system_info;
    std::unordered_map<mo::TypeInfo, std::unique_ptr<mo::ISingletonContainer>> m_singletons;
    std::function<mo::AllocatorPtr_t()> m_allocator_constructor;
}; // struct SystemTable

//////////////////////////////////////////////////////////////////////////////
///                              Implementation
//////////////////////////////////////////////////////////////////////////////

void SystemTable::staticDispatchToSystemTable(std::function<void(SystemTable*)>&& func)
{
    auto instance = PerModuleInterface::GetInstance();
    if (instance)
    {
        auto table = instance->GetSystemTable();

        if (table)
        {
            func(table);
        }
        else
        {
            instance->AddDelayInitFunction(func);
        }
    }
}

namespace mo
{

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
            TSingletonContainer<T>::ptr = m_ptr.get();
        }
        SharingContainer(const std::shared_ptr<T>& ptr_)
            : m_ptr(ptr_)
        {
            TSingletonContainer<T>::ptr = m_ptr.get();
        }

        std::shared_ptr<T> ptr()
        {
            return m_ptr;
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
                mo::getDefaultLogger().info("Creating new {} singleton instance {} in system table ({})",
                                            mo::TypeTable::instance(table).typeToName(mo::TypeInfo(typeid(U))),
                                            static_cast<const void*>(ptr),
                                            static_cast<const void*>(table));
            }
        }
        return ptr;
    }

    template <class T, class U = T>
    std::shared_ptr<T> sharedSingleton(SystemTable* table)
    {
        std::shared_ptr<T> ptr = nullptr;
        if (table)
        {
            ptr = table->getSharedSingleton<T>();
            if (ptr == nullptr)
            {
                ptr = table->setSingleton(std::make_shared<U>());
                mo::getDefaultLogger().info("Creating new shared {} singleton instance {} in system table ({})",
                                            mo::TypeTable::instance(table).typeToName(mo::TypeInfo(typeid(U))),
                                            static_cast<const void*>(ptr.get()),
                                            static_cast<const void*>(table));
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
}

template <class T>
T* SystemTable::getSingleton()
{
    auto itr = m_singletons.find(mo::TypeInfo(typeid(T)));
    if (itr != m_singletons.end())
    {
        auto container = static_cast<mo::TSingletonContainer<T>*>(itr->second.get());
        if (container)
        {
            return container->ptr;
        }
    }
    return nullptr;
}

template <class T>
std::shared_ptr<T> SystemTable::getSharedSingleton()
{
    auto itr = m_singletons.find(mo::TypeInfo(typeid(T)));
    if (itr != m_singletons.end())
    {
        auto container = static_cast<mo::SharingContainer<T, void>*>(itr->second.get());
        if (container)
        {
            return container->ptr();
        }
    }
    return nullptr;
}

// Owning
template <typename T>
T* SystemTable::setSingleton(std::unique_ptr<T>&& singleton)
{
    std::unique_ptr<mo::OwningContainer<T>> owner(new mo::OwningContainer<T>(std::move(singleton)));
    T* ptr = owner->ptr;
    m_singletons[mo::TypeInfo(typeid(T))] = std::move(owner);
    return ptr;
}

// Shared
template <typename T>
T* SystemTable::setSingleton(const rcc::shared_ptr<T>& singleton)
{
    std::unique_ptr<mo::SharingContainer<T, void>> owner(new mo::SharingContainer<T, void>(singleton));
    T* ptr = owner->ptr;
    m_singletons[mo::TypeInfo(typeid(T))] = std::move(owner);
    return ptr;
}

template <typename T>
std::shared_ptr<T> SystemTable::setSingleton(const std::shared_ptr<T>& singleton)
{
    m_singletons[mo::TypeInfo(typeid(T))] =
        std::unique_ptr<mo::SharingContainer<T, void>>(new mo::SharingContainer<T, void>(singleton));
    return singleton;
}

// Non owning
template <typename T>
T* SystemTable::setSingleton(T* ptr)
{
    std::unique_ptr<mo::NonOwningContainer<T>> owner(new mo::NonOwningContainer<T>(ptr));
    m_singletons[mo::TypeInfo(typeid(T))] = std::move(owner);
    return ptr;
}

template <typename T>
void SystemTable::deleteSingleton()
{
    deleteSingleton(typeid(T));
}
