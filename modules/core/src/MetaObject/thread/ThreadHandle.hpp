#pragma once
#include <MetaObject/core/detail/Forward.hpp>
#include <MetaObject/detail/Export.hpp>
#include <functional>
#include <memory>

namespace mo {
class Thread;
class ThreadPool;
class Context;
class ISlot;
class Connection;
template <class T>
class TSlot;

/*!
     * \brief The ThreadHandle class is used to wrap the mo::Thread in such a way that the implementation an be hidden from nvcc
     *        since nvcc has compile errors with many boost/thread headers.  Since contexts contain cuda allocators, and it is expected that 
     *        the allocator out lives any created objects (OpenCV doesn't do reference counting on allocators); threads are recycled to the thread
     *        pool so that when a thread is no longer being used its allocator can still live.  Furthermore this simplifies cuda callbacks by allowing
     *        a thread to finish any asynchronous work after the work enqueing object is deleted.
     */
class MO_EXPORTS ThreadHandle {
public:
    ThreadHandle();
    ThreadHandle(const ThreadHandle& other);
    ThreadHandle(ThreadHandle&& other);

    ~ThreadHandle();

    ThreadHandle& operator=(ThreadHandle&& other);
    ThreadHandle& operator=(const ThreadHandle& other);

    ContextPtr_t       getContext() const;
    size_t             getId() const;
    bool               getIsRunning() const;
    const std::string& getThreadName() const;
    bool               isOnThread() const;

    void pushEventQueue(const std::function<void(void)>& f); // Events must be handled on the enqueued thread
    void pushWork(const std::function<void(void)>& f); // Work can be stolen and can exist on any thread
    void start();
    void stop();

    void setExitCallback(const std::function<void(void)>& f);
    void setStartCallback(const std::function<void(void)>& f);
    void setThreadName(const std::string& name);
    std::shared_ptr<Connection> setInnerLoop(TSlot<int(void)>* slot);

protected:
    friend class ThreadPool;
    ThreadHandle(Thread* thread, int* ref_count);
    Thread* _thread;
    int*    _ref_count;
    void    decrement();
    void    increment();
};
}
