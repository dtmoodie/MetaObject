#pragma once
#include <MetaObject/core/detail/forward.hpp>

#include <cstdint>
#include <memory>

namespace mo
{
    class Thread;
    class ThreadHandle
    {
      public:
        ThreadHandle(const std::shared_ptr<Thread>& thread);

        IAsyncStreamPtr_t asyncStream(const Duration timeout = 5 * second) const;
        size_t threadId() const;
        bool isOnThread() const;
        const std::string& threadName() const;

        void setName(const std::string& name);

      private:
        std::shared_ptr<Thread> m_thread;
    };
} // namespace mo
