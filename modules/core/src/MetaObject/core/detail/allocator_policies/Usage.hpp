#pragma once
#include "../Allocator.hpp"
#include <MetaObject/logging/logging.hpp>
#include <MetaObject/logging/logging_macros.hpp>

#include <boost/stacktrace.hpp>

#include <cstdint>
#include <sstream>
#include <unordered_map>
namespace mo
{
    // Keeps track of how much memory it has allocated
    template <class Allocator>
    struct UsagePolicy : public Allocator
    {
        uint8_t* allocate(size_t num_bytes, size_t elem_size) override;

        void deallocate(uint8_t* ptr, size_t num_bytes) override;

        size_t usage() const;

        void printUsage(std::ostream& os) const;

      protected:
        size_t m_usage = 0;
        size_t m_warn_at = 1024UL * 1024UL * 1024UL * 2UL;
        std::unordered_map<const void*, std::string> m_stack_traces;
    };

    //////////////////////////////////////////////////////////////////////////////////////////////////////
    ///                            Implementation
    //////////////////////////////////////////////////////////////////////////////////////////////////////

    template <class Allocator>
    uint8_t* UsagePolicy<Allocator>::allocate(const size_t num_bytes, const size_t elem_size)
    {
        auto ptr = Allocator::allocate(num_bytes, elem_size);
        if (ptr)
        {
            m_usage += num_bytes;

            std::stringstream ss;
            boost::stacktrace::stacktrace st;
            ss << st << '\n';
            ss << num_bytes;
            m_stack_traces[ptr] = ss.str();
            if (m_usage > m_warn_at)
            {
                printUsage(std::cout);
            }
        }
        return ptr;
    }

    template <class Allocator>
    void UsagePolicy<Allocator>::printUsage(std::ostream& os) const
    {
        for (const auto& itr : m_stack_traces)
        {
            os << itr.second << std::endl;
        }
    }

    template <class Allocator>
    void UsagePolicy<Allocator>::deallocate(uint8_t* ptr, const size_t num_bytes)
    {
        Allocator::deallocate(ptr, num_bytes);
        m_usage -= num_bytes;
        auto itr = m_stack_traces.find(ptr);
        MO_ASSERT(itr != m_stack_traces.end());
        m_stack_traces.erase(itr);
    }

    template <class Allocator>
    size_t UsagePolicy<Allocator>::usage() const
    {
        return m_usage;
    }
} // namespace mo
