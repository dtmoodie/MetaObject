#include <MetaObject/params/MetaParam.hpp>
#include "MetaObject/metaparams/MetaParamsInclude.hpp"

#include <MetaObject/params/ITAccessibleParam.hpp>
#include "MetaObject/params/Types.hpp"
#include <MetaObject/params/AccessToken.hpp>
#include <boost/lexical_cast.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>

namespace mo{
namespace IO{
namespace Text{
namespace imp{

bool DeSerialize(ITAccessibleParam<EnumParam>* param, std::stringstream& ss){
    auto token = param->access();
    token().values.clear();
    (token)().enumerations.clear();
    std::string size;
    std::getline(ss, size, '[');
    if (size.size()){
        size_t size_ = boost::lexical_cast<size_t>(size);
        (token)().values.reserve(size_);
        (token)().enumerations.reserve(size_);
    }
    std::string enumeration;
    int value;
    char ch;
    while( ss >> enumeration >> ch >> value){
        (token)().addEnum(value, enumeration);
        ss >> ch;
    }
    return true;
}

bool Serialize(ITAccessibleParam<EnumParam>* param, std::stringstream& ss){
    auto token = param->access();
    (token)().enumerations.size();
    ss << "[";
    for(int i = 0; i < (token)().enumerations.size(); ++i){
        if(i != 0)
            ss << ", ";
        ss << (token)().enumerations[i] << ":" << (token)().values[i];
    }
    ss << "]";
    return true;
}

}
} // namespace Text
} // namespace IO
} // namespace mo

#ifdef MO_EXPORTS
#undef MO_EXPORTS
#endif
#if (defined WIN32 || defined _WIN32 || defined WINCE || defined __CYGWIN__)
#  define MO_EXPORTS __declspec(dllexport)
#elif defined __GNUC__ && __GNUC__ >= 4
#  define MO_EXPORTS __attribute__ ((visibility ("default")))
#else
#  define MO_EXPORTS
#endif

#include "MetaObject/params/detail/MetaParamImpl.hpp"

namespace mo
{
    std::ostream& operator<<(std::ostream& os, const mo::EnumParam& obj)
    {
        ASSERT_SERIALIZABLE(EnumParam);
        for(size_t i = 0; i < obj.enumerations.size(); ++i)
        {
            if(i == obj.currentSelection)
            {
                os << '>' << obj.enumerations[i] << "<, ";
            }else
            {
                os << obj.enumerations[i] << ", ";
            }
        }
        return os;
    }
}

using namespace mo;

template<class AR>
void EnumParam::serialize(AR& ar)
{
    ar(CEREAL_NVP(enumerations), CEREAL_NVP(values), CEREAL_NVP(currentSelection));
}

namespace cereal
{

}

INSTANTIATE_META_PARAM(ReadFile);
INSTANTIATE_META_PARAM(WriteFile);
INSTANTIATE_META_PARAM(ReadDirectory);
INSTANTIATE_META_PARAM(WriteDirectory);
INSTANTIATE_META_PARAM(EnumParam);
