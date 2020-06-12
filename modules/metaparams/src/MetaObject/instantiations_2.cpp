#include "common.hpp"

#include <MetaObject/runtime_reflection/visitor_traits/string.hpp>
#include <MetaObject/runtime_reflection/visitor_traits/vector.hpp>

#ifdef MO_EXPORTS
#undef MO_EXPORTS
#endif
#if (defined WIN32 || defined _WIN32 || defined WINCE || defined __CYGWIN__)
#define MO_EXPORTS __declspec(dllexport)
#elif defined __GNUC__ && __GNUC__ >= 4
#define MO_EXPORTS __attribute__((visibility("default")))
#else
#define MO_EXPORTS
#endif

#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>

namespace mo
{
    namespace metaparams
    {
        void instantiateVectors(SystemTable*)
        {
            registerTrait<std::vector<int>>();
            registerTrait<std::vector<unsigned short>>();
            registerTrait<std::vector<unsigned int>>();
            registerTrait<std::vector<char>>();
            registerTrait<std::vector<unsigned char>>();
            registerTrait<std::vector<float>>();
            registerTrait<std::vector<double>>();
            registerTrait<std::vector<std::string>>();
        }
    } // namespace metaparams
} // namespace mo
