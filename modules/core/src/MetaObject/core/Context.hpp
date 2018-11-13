#pragma once
#include "MetaObject/core.hpp"
#include "MetaObject/detail/TypeInfo.hpp"
#include <MetaObject/core/metaobject_config.hpp>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace mo
{
    class Context;
    class Thread;
    class EventToken;
    namespace cuda
    {
        struct Context;
        struct CvContext;
    }

    struct ContextConstructor;

    struct MO_EXPORTS ContextFactory
    {
        using Ptr = std::shared_ptr<Context>;
        static ContextFactory* instance();

        void registerConstructor(ContextConstructor* ctr);

        Ptr create(const std::string& name, int device_id = 0, int device_priority = 5, int thread_priority = 0);

      private:
        std::vector<ContextConstructor*> m_ctrs;
    }; // struct mo::ContextFactory

    class Allocator;
    class MO_EXPORTS IContext
    {
      public:
        using Ptr = std::shared_ptr<IContext>;

        /*!
         * \brief creates a Context based on the underlying hardware
         * \param name of the context
         * \param priority of the created cuda stream
         * \return shared ptr to created context
         */
        static Ptr
        create(const std::string& name = "", int device_id = 0, int cuda_priority = 5, int thread_priority = 0);
        static IContext* getCurrent();
        static void setCurrent(IContext* ctx);

        virtual ~IContext();

        virtual void pushWork(std::function<void(void)>&& work) = 0;
        virtual void pushEvent(EventToken&&) = 0;

        virtual void setName(const std::string& name) = 0;

        virtual std::string name() const = 0;
        virtual size_t threadId() const = 0;
        virtual bool isDeviceContext() const = 0;
        virtual size_t processId() const = 0;
        virtual std::shared_ptr<Allocator> allocator() const = 0;

        virtual void setEventHandle(std::function<void(EventToken&&)>&& event_handler) = 0;
        virtual void setWorkHandler(std::function<void(std::function<void(void)>)>&& work_handler) = 0;
    }; // class mo::IContext

    class MO_EXPORTS Context : virtual public IContext
    {
      public:
        using Ptr = std::shared_ptr<Context>;

        /*!
         * \brief creates a Context based on the underlying hardware
         * \param name of the context
         * \param priority of the created cuda stream
         * \return shared ptr to created context
         */
        static Ptr
        create(const std::string& name = "", int cuda_priority = 5, int thread_priority = 0, int device_id = 0);

        Context();
        virtual ~Context();

        void setName(const std::string& name) override;

        void pushWork(std::function<void(void)>&& work) override;
        void pushEvent(EventToken&&) override;
        std::string name() const override;
        size_t threadId() const override;
        bool isDeviceContext() const override;
        size_t processId() const override;
        std::shared_ptr<Allocator> allocator() const override;

        void setEventHandle(std::function<void(EventToken&&)>&& event_handler) override;
        void setWorkHandler(std::function<void(std::function<void(void)>)>&& work_handler) override;

      protected:
        Context(TypeInfo concrete_type);

        virtual void setDeviceId(const int device_id);
        virtual void setAllocator(const std::shared_ptr<Allocator>& allocator);

      private:
        std::string m_name;
        size_t m_process_id = 0;
        size_t m_thread_id = 0;
        int m_device_id = -1;
        std::string m_host_name;
        std::shared_ptr<Allocator> m_allocator;
        // Type of derived class, used for type switch
        mo::TypeInfo m_context_type;

        std::function<void(std::function<void(void)>)> m_work_handler;
        std::function<void(EventToken&&)> m_event_handler;
    }; // class mo::IContext

} // namespace mo
