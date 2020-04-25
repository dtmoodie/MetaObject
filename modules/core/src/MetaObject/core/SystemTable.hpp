#pragma once
#include "IObjectTable.hpp"

#include <MetaObject/core/TypeTable.hpp>
#include <MetaObject/core/detail/Allocator.hpp>
#include <MetaObject/core/detail/HelperMacros.hpp>

#include <MetaObject/core/detail/forward.hpp>
#include <MetaObject/detail/Export.hpp>

#include <MetaObject/detail/defines.hpp>
#include <MetaObject/logging/logging.hpp>

#include <RuntimeObjectSystem/ObjectInterfacePerModule.h>
#include <RuntimeObjectSystem/shared_ptr.hpp>

#include <memory>
#include <unordered_map>

#ifdef GetModuleFileName
#undef GetModuleFileName
#endif

struct SystemInfo
{
    bool have_cuda = false;
};

namespace mo
{
    class MetaObjectFactory;
}

struct MO_EXPORTS SystemTable : virtual private mo::IObjectTable
{
  public:
    using Ptr_t = std::shared_ptr<SystemTable>;

    SystemTable(const SystemTable& other) = delete;
    SystemTable(SystemTable&& other) = delete;
    SystemTable& operator=(const SystemTable& other) = delete;
    SystemTable& operator=(SystemTable&& other) = delete;

    MO_INLINE static std::shared_ptr<SystemTable> instance();

    virtual ~SystemTable();

    template <class T, class U = T>
    mo::SharedPtrType<T> getSingleton();

    template <class T>
    mo::SharedPtrType<T> getSingletonOptional();

    // Shared
    template <typename PTR>
    void setSingleton(PTR&& singleton);

    void deleteSingleton(mo::TypeInfo type);

    template <typename T>
    void deleteSingleton();

    // Members
    void setAllocatorConstructor(std::function<mo::AllocatorPtr_t()>&& ctr);
    mo::AllocatorPtr_t createAllocator() const;

    mo::AllocatorPtr_t getDefaultAllocator();
    void setDefaultAllocator(mo::AllocatorPtr_t);

    static MO_INLINE void dispatchToSystemTable(std::function<void(SystemTable*)>&& func);

    const std::shared_ptr<spdlog::logger>& getDefaultLogger();
    void setDefaultLogger(const std::shared_ptr<spdlog::logger>& logger);

    MO_INLINE void registerModule();

  protected:
  private:
    static std::shared_ptr<SystemTable> instanceImpl();
    SystemTable();
    mo::AllocatorPtr_t m_default_allocator;
    SystemInfo m_system_info;
    std::unordered_map<mo::TypeInfo, IObjectContainer::Ptr_t> m_singletons;
    std::function<mo::AllocatorPtr_t()> m_allocator_constructor;

    IObjectContainer* getObjectContainer(mo::TypeInfo) const override;
    void setObjectContainer(mo::TypeInfo, IObjectContainer::Ptr_t&&) override;
    std::shared_ptr<spdlog::logger> m_logger;
}; // struct SystemTable

//////////////////////////////////////////////////////////////////////////////
///                              Implementation
//////////////////////////////////////////////////////////////////////////////

std::shared_ptr<SystemTable> SystemTable::instance()
{
    auto inst = instanceImpl();
    auto module = PerModuleInterface::GetInstance();
    module->SetSystemTable(inst.get());
    return inst;
}

void SystemTable::dispatchToSystemTable(std::function<void(SystemTable*)>&& func)
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
    DEFINE_HAS_STATIC_FUNCTION(HasSystemTableInstance, instance, SharedPtrType<U>, SystemTable*);

    template <class T>
    typename std::enable_if<HasSystemTableInstance<T>::value, SharedPtrType<T>>::type invokeInstance(SystemTable* table)
    {
        return T::instance(table);
    }

    template <class T>
    typename std::enable_if<!HasSystemTableInstance<T>::value, SharedPtrType<T>>::type
    invokeInstance(SystemTable* table)
    {
        return table->getSingleton<T>();
    }

    template <class T>
    SharedPtrType<T> singleton()
    {
        SharedPtrType<T> ptr;
        auto module = PerModuleInterface::GetInstance();
        MO_ASSERT(module);
        auto table = module->GetSystemTable();
        if (table == nullptr)
        {
            auto inst = SystemTable::instance();
            if (inst)
            {
                const std::string name = module->GetModuleFileName();
                table = inst.get();
                MO_LOG(
                    warn,
                    "System table not setup for this module '{}', using less prefered SystemTable::instance to get the "
                    "system table, got instance at {}",
                    name,
                    static_cast<void*>(table));
                module->SetSystemTable(table);
            }
        }
        MO_ASSERT(table != nullptr);
        return invokeInstance<T>(table);
    }
} // namespace mo

template <class T, class U>
mo::SharedPtrType<T> SystemTable::getSingleton()
{
    registerModule();
    return mo::IObjectTable::getObject<T, U>();
}

template <class T>
mo::SharedPtrType<T> SystemTable::getSingletonOptional()
{
    registerModule();
    return mo::IObjectTable::getObjectOptional<T>();
}

void SystemTable::registerModule()
{
    auto module = PerModuleInterface::GetInstance();
    auto table_ = module->GetSystemTable();
    if (table_)
    {
        MO_ASSERT(this == table_);
    }
    else
    {
        module->SetSystemTable(this);
    }
}

template <typename PTR>
void SystemTable::setSingleton(PTR&& singleton)
{
    mo::IObjectTable::setObject(std::move(singleton));
}

template <typename T>
void SystemTable::deleteSingleton()
{
    deleteSingleton(mo::TypeInfo::create<T>());
}
