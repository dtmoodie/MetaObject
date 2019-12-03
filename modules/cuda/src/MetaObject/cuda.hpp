#ifndef MO_CUDA_HPP
#define MO_CUDA_HPP

#define METAOBJECT_MODULE "cuda"
#include <MetaObject/detail/Export.hpp>
#undef METAOBJECT_MODULE

struct SystemTable;
namespace mo
{
    namespace cuda
    {
        void MO_EXPORTS init(SystemTable* table);
    }
} // namespace mo
#endif // MO_CUDA_HPP
