#pragma once
#include <MetaObject/core/detail/Forward.hpp>
#include <MetaObject/detail/Export.hpp>

#include <cstring>
#include <functional>
#include <memory>
namespace mo
{
    class Thread;
    class ThreadPool;
    class Context;
    class ISlot;
    class Connection;

    // TODO not needed once we integrate cache engine
    // borrowed from https://gitlab.com/dtmoodie/cache-engine/blob/WIP/include/ce/detail/hash.hpp
    inline uint64_t combineHash(uint64_t seed, const uint64_t hash)
    {
        seed ^= hash + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }

    template <class T>
    inline uint64_t generateHash(const T& val)
    {
        return std::hash<T>{}(val);
    }

    template <class T>
    inline uint64_t generateHash(const T* val)
    {
        return std::hash<const void*>{}(val);
    }

    template <class T, std::size_t N>
    inline uint64_t generateHash(const std::array<T, N>& array)
    {
        uint64_t hash = 0;
        for (uint64_t i = 0; i < N; ++i)
        {
            hash = combineHash(hash, generateHash(array[i]));
        }
        return hash;
    }

    template <class T, class R, class... Args>
    uint64_t hashFptr(R (T::*fptr)(Args...))
    {
        const char* ptrptr = static_cast<const char*>(static_cast<const void*>(&fptr));
        std::array<char, sizeof(fptr)> val;
        std::memcpy(&val[0], ptrptr, sizeof(fptr));
        return generateHash(val);
    }

    // Add a level of indirection such that boost/thread header files do not need to be viewed by nvcc
    // this is a work around for older versions of nvcc that would implode on including certain boost headers
    class MO_EXPORTS ThreadHandle
    {
      public:
        ThreadHandle(const std::shared_ptr<Thread>& thread);

        ContextPtr_t context() const;
        size_t threadId() const;

        const std::string& threadName() const;
        bool isOnThread() const;

        // Events must be handled on the enqueued thread
        bool pushEventQueue(std::function<void(void)>&& f, const uint64_t id = 0);

        template <class T, class... Args>
        bool pushEventQueue(T* obj, void (T::*fptr)(Args...), Args&&... args)
        {
            return pushEventQueue([obj, fptr, args...]() { (*obj.*fptr)(args...); },
                                  combineHash(generateHash(obj), hashFptr(fptr)));
        }
        // Work can be stolen and can exist on any thread
        bool pushWork(std::function<void(void)>&& f);

        void setExitCallback(std::function<void(void)>&& f);
        void setThreadName(const std::string& name);

      private:
        std::shared_ptr<Thread> m_thread;
    };
}
