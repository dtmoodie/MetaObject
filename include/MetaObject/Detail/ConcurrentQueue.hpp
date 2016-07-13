#pragma once
#include "MetaObject/Logging/Log.hpp"

#include <boost/thread/recursive_mutex.hpp>
#include <queue>
#include <boost/thread.hpp>

template<typename T> void cleanup(T& ptr, typename std::enable_if< std::is_array<T>::value>::type* = 0) { /*delete[] ptr;*/ }
template<typename T> void cleanup(T& ptr, typename std::enable_if< std::is_pointer<T>::value && !std::is_array<T>::value>::type* = 0) { delete ptr; }
template<typename T> void cleanup(T& ptr, typename std::enable_if<!std::is_pointer<T>::value && !std::is_array<T>::value>::type* = 0) { return; }

template<typename Data>
class ConcurrentNotifier
{
private:
    boost::condition_variable the_condition_variable;
    std::vector<Data> the_data;
    mutable boost::mutex the_mutex;
public:
    void wait_for_data()
    {
        boost::mutex::scoped_lock lock(the_mutex);
        while (the_data.empty())
        {
            the_condition_variable.wait(lock);
        }
    }
    void wait_push(Data const& data)
    {
        boost::mutex::scoped_lock lock(the_mutex);
        while (!the_data.empty()) // Wait till the consumer pulls data from the queue
        {
            the_condition_variable.wait(lock);
        }
        the_data.push_back(data);
    }

    void push(Data const& data)
    {
        boost::mutex::scoped_lock lock(the_mutex);
        bool const was_empty = the_data.empty();
        if (the_data.size())
            the_data[0] = data;
        else
            the_data.push_back(data);

        lock.unlock(); // unlock the mutex

        if (was_empty)
        {
            the_condition_variable.notify_one();
        }
    }
    void wait_and_pop(Data& popped_value)
    {
        boost::mutex::scoped_lock lock(the_mutex);
        while (the_data.empty())
        {
            the_condition_variable.wait(lock);
        }

        popped_value = the_data[0];
        the_data.clear();
    }
    bool try_pop(Data& popped_value)
    {
        boost::mutex::scoped_lock lock(the_mutex);
        if (the_data.empty())
            return false;
        popped_value = the_data[0];
        the_data.clear();
        return true;
    }

    size_t size()
    {
        boost::mutex::scoped_lock lock(the_mutex);
        return the_data.size();
    }
    void clear()
    {
        boost::mutex::scoped_lock lock(the_mutex);
        the_data.clear();
    }
};

template<typename Data, typename Container = std::deque<Data>>
class ConcurrentQueue: public std::queue<Data, Container>
{
private:
    typedef std::queue<Data, std::deque<Data>> Super;
    boost::condition_variable the_condition_variable;
    //std::queue<Data> the_queue;
    mutable boost::mutex the_mutex;
public:
    typedef typename Container::iterator iterator;
    typedef typename Container::const_iterator const_iterator;
    void wait_for_data()
    {
        boost::mutex::scoped_lock lock(the_mutex);
        while (Super::empty())
        {
            the_condition_variable.wait(lock);
        }
    }
    void wait_push(Data const& data)
    {
        boost::mutex::scoped_lock lock(the_mutex);
        while (!Super::empty()) // Wait till the consumer pulls data from the queue
        {
            the_condition_variable.wait(lock);
        }
        Super::push_back(data);
    }
    void push(Data const& data)
    {
        boost::mutex::scoped_lock lock(the_mutex);
        bool const was_empty = Super::empty();
        Super::push(data);

        lock.unlock(); // unlock the mutex

        if (was_empty)
        {
            the_condition_variable.notify_one();
        }
    }
    void wait_and_pop(Data& popped_value)
    {
        boost::mutex::scoped_lock lock(the_mutex);
        while (Super::empty())
        {
            the_condition_variable.wait(lock);
        }

        popped_value = Super::front();
        Super::pop();
    }
    bool try_pop(Data& popped_value)
    {
        boost::mutex::scoped_lock lock(the_mutex);
        if (Super::empty())
            return false;
        popped_value = Super::front();
        Super::pop();
        return true;
    }

    size_t size()
    {
        boost::mutex::scoped_lock lock(the_mutex);
        return Super::size();
    }
    void clear()
    {
        boost::mutex::scoped_lock lock(the_mutex);
        Super::c.clear();
    }
    iterator begin() { return this->c.begin(); }
    iterator end() { return this->c.end(); }
    const_iterator begin() const { return this->c.begin(); }
    const_iterator end() const { return this->c.end(); }
    iterator erase(iterator item)
    {
        boost::mutex::scoped_lock lock(the_mutex);
        return this->c.erase(item);
    }
};