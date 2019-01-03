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
        static ILoadVisitor& load(ILoadVisitor& visitor, ArrayAdapter<T, N>* val, const std::string&, const size_t)
        {
            visitor(val->ptr, "data", N);
        }

        static ISaveVisitor& save(ISaveVisitor& visitor, const ArrayAdapter<T, N>* val, const std::string&, const size_t)
        {
            visitor(val->ptr, "data", N);
            return visitor;
        }

        static void visit(StaticVisitor& visitor, const std::string& name, const size_t cnt)
        {
            visitor.template visit<T>("data", N);
        }
    };

    template<class T, size_t ROWS, size_t COLS>
    struct Visit<MatrixAdapter<T, ROWS, COLS>>
    {
        static ILoadVisitor&
        load(ILoadVisitor& visitor, MatrixAdapter<T, ROWS, COLS>* val, const std::string&, const size_t)
        {
            visitor(val->ptr, "data", ROWS * COLS);
            return visitor;
        }

        static ISaveVisitor&
        save(ISaveVisitor& visitor, const MatrixAdapter<T, ROWS, COLS>* val, const std::string&, const size_t)
        {
            visitor(val->ptr, "data", ROWS * COLS);
            return visitor;
        }

        static void
        visit(StaticVisitor& visitor, const std::string& name, const size_t cnt)
        {
            visitor.template visit<T>("data", ROWS * COLS);
        }
    };
}

#endif // MO_VISITATION_ARRAY_ADAPTER_HPP
