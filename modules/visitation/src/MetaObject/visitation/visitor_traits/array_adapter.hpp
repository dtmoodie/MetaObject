#ifndef MO_VISITATION_ARRAY_ADAPTER_HPP
#define MO_VISITATION_ARRAY_ADAPTER_HPP
#include <MetaObject/types/ArrayAdapater.hpp>
#include <MetaObject/visitation/IDynamicVisitor.hpp>

namespace mo
{
    template <class T, size_t N>
    IWriteVisitor& visit(IWriteVisitor& visitor, const ArrayAdapter<T, N>* val, const std::string&, const size_t)
    {
        const uint32_t n = N;
        visitor(&n, std::string("len"));
        visitor(val->ptr, "data", n);
        return visitor;
    }

    template <class T, size_t ROWS, size_t COLS>
    IWriteVisitor&
    visit(IWriteVisitor& visitor, const MatrixAdapter<T, ROWS, COLS>* val, const std::string&, const size_t)
    {
        const uint32_t rows = ROWS;
        const uint32_t cols = COLS;
        visitor(&rows, std::string("rows"));
        visitor(&cols, std::string("cols"));
        visitor(val->ptr, "data", ROWS * COLS);
        return visitor;
    }
}

#endif // MO_VISITATION_ARRAY_ADAPTER_HPP
