#ifndef MO_VISITATION_ARRAY_ADAPTER_HPP
#define MO_VISITATION_ARRAY_ADAPTER_HPP
#include <MetaObject/types/ArrayAdapater.hpp>
#include <MetaObject/visitation/IDynamicVisitor.hpp>
#include <MetaObject/logging/logging.hpp>

namespace mo
{
    template<class T, size_t N>
    struct Visit<ArrayAdapter<T, N>>
    {
        static IReadVisitor& read(IReadVisitor& visitor, ArrayAdapter<T, N>* val, const std::string&, const size_t)
        {
            visitor(val->ptr, "data", N);
        }

        static IWriteVisitor& write(IWriteVisitor& visitor, const ArrayAdapter<T, N>* val, const std::string&, const size_t)
        {
            visitor(val->ptr, "data", N);
            return visitor;
        }

        static void visit(StaticVisitor&, const std::string& name, const size_t cnt)
        {

        }
    };

    template<class T, size_t ROWS, size_t COLS>
    struct Visit<MatrixAdapter<T, ROWS, COLS>>
    {
        static IReadVisitor&
        read(IReadVisitor& visitor, MatrixAdapter<T, ROWS, COLS>* val, const std::string&, const size_t)
        {
            visitor(val->ptr, "data", ROWS * COLS);
            return visitor;
        }

        static IWriteVisitor&
        write(IWriteVisitor& visitor, const MatrixAdapter<T, ROWS, COLS>* val, const std::string&, const size_t)
        {
            visitor(val->ptr, "data", ROWS * COLS);
            return visitor;
        }

        static void
        visit(StaticVisitor& visitor, const std::string& name, const size_t cnt)
        {

        }
    };
}

#endif // MO_VISITATION_ARRAY_ADAPTER_HPP
