#ifndef MO_CUDA_ERRORS_HPP
#define MO_CUDA_ERRORS_HPP
#include <MetaObject/logging/logging.hpp>
#include <MetaObject/logging/logging_macros.hpp>

#include <driver_types.h>
namespace std
{
    ostream& operator<<(ostream& os, const std::vector<const char*>& vec);
} // namespace std

namespace mo
{
    namespace cuda
    {
        template <class... FARGS, class... ARGS>
        void checkCudaError(cudaError_t (*func)(FARGS...), const std::vector<const char*>& expression, ARGS&&... args)
        {
            const cudaError_t error = func(std::forward<ARGS>(args)...);
            if (error != cudaSuccess)
            {
                THROW(error, "cuda error \"{}\" when executing {}", error, expression);
            }
        }
    } // namespace cuda
} // namespace mo

#include <boost/preprocessor.hpp>

#define TOKENIZE_ARGUMENT(r, unused, idx, elem) BOOST_PP_COMMA_IF(idx) BOOST_PP_STRINGIZE(elem)
#define TOKENIZE_ARGS(...) BOOST_PP_SEQ_FOR_EACH_I(TOKENIZE_ARGUMENT, "", BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))

#define CHECK_CUDA_ERROR(foo, ...) mo::cuda::checkCudaError(foo, {#foo, TOKENIZE_ARGS(__VA_ARGS__)}, __VA_ARGS__)

#endif // MO_CUDA_ERRORS_HPP
