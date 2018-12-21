#ifndef MO_VISITATION_ARRAY_ADAPTER_HPP
#define MO_VISITATION_ARRAY_ADAPTER_HPP
#include <MetaObject/types/ArrayAdapater.hpp>
#include <MetaObject/visitation/IDynamicVisitor.hpp>
#include <MetaObject/logging/logging.hpp>

namespace mo
{
    template<class T, size_t N>
    IReadVisitor& read(IReadVisitor& visitor, ArrayAdapter<T, N>* val, const std::string&, const size_t)
    {
        uint32_t n = 0;
        visitor(&n, "len");
        MO_ASSERT_EQ(n, N);
        visitor(val->ptr, "data", n);
    }

    template <class T, size_t N>
    IWriteVisitor& write(IWriteVisitor& visitor, const ArrayAdapter<T, N>* val, const std::string&, const size_t)
    {
        const uint32_t n = N;
        visitor(&n, std::string("len"));
        visitor(val->ptr, "data", n);
        return visitor;
    }

    template<class T, size_t ROWS, size_t COLS>
    IReadVisitor&
    read(IReadVisitor& visitor, MatrixAdapter<T, ROWS, COLS>* val, const std::string&, const size_t)
    {
        visitor(val->ptr, "data", ROWS * COLS);
        return visitor;
    }

    template <class T, size_t ROWS, size_t COLS>
    IWriteVisitor&
    write(IWriteVisitor& visitor, const MatrixAdapter<T, ROWS, COLS>* val, const std::string&, const size_t)
    {
        visitor(val->ptr, "data", ROWS * COLS);
        return visitor;
    }
}

#endif // MO_VISITATION_ARRAY_ADAPTER_HPP
