#include <MetaObject/params/MetaParam.hpp>
#ifdef HAVE_QT
#include "MetaObject/params/ui/Qt/OpenCV.hpp"
#include "MetaObject/params/ui/Qt/Containers.hpp"
#include "MetaObject/params/ui/Qt/TParamProxy.hpp"
#endif
#include "MetaObject/params/buffers/CircularBuffer.hpp"
#include "MetaObject/params/buffers/StreamBuffer.hpp"
#include "MetaObject/params/buffers/Map.hpp"
#include "MetaObject/serialization/CerealPolicy.hpp"
#include "MetaObject/serialization/TextPolicy.hpp"
#include "MetaObject/params/Types.hpp"
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

#include <cereal/types/vector.hpp>
#include <cereal/types/string.hpp>

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

namespace cereal{
    template<class Archive> void load(Archive& ar, mo::ReadFile& m){
        std::string path;
        ar(path);
        m = path;
    }
    template<class Archive> void save(Archive& ar, mo::ReadFile const & m){
        std::string path = m.string();
        ar(path);
    }
    template<class Archive> void load(Archive& ar, mo::WriteFile& m){
        std::string path;
        ar(path);
        m = path;
    }
    template<class Archive> void save(Archive& ar, mo::WriteFile const& m){
        std::string path = m.string();
        ar(path);
    }
    template<class Archive> void load(Archive& ar, mo::ReadDirectory& m){
        std::string path;
        ar(path);
        m = mo::ReadDirectory(path);
    }
    template<class Archive> void save(Archive& ar, mo::ReadDirectory const& m){
        std::string path = m.string();
        ar(path);
    }
    template<class Archive> void load(Archive& ar, mo::WriteDirectory& m){
        std::string path;
        ar(path);
        m = path;
    }
    template<class Archive> void save(Archive& ar, mo::WriteDirectory const& m){
        std::string path = m.string();
        ar(path);
    }
}
using namespace mo;
template<class AR> void EnumParam::serialize(AR& ar){
    ar(CEREAL_NVP(enumerations), CEREAL_NVP(values), CEREAL_NVP(currentSelection));
}
INSTANTIATE_META_Param(ReadFile);
INSTANTIATE_META_Param(WriteFile);
INSTANTIATE_META_Param(ReadDirectory);
INSTANTIATE_META_Param(WriteDirectory);
INSTANTIATE_META_Param(EnumParam);
