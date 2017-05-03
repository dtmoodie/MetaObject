#include "MetaObject/Params/MetaParam.hpp"
#include "MetaObject/Params/UI/Qt/OpenCV.hpp"
#include "MetaObject/Params/UI/Qt/Containers.hpp"
#include "MetaObject/Params/UI/Qt/TParamProxy.hpp"
#include "MetaObject/Params/Buffers/CircularBuffer.hpp"
#include "MetaObject/Params/Buffers/StreamBuffer.hpp"
#include "MetaObject/Params/Buffers/Map.hpp"
#include "MetaObject/Params/IO/CerealPolicy.hpp"
#include "MetaObject/Params/IO/TextPolicy.hpp"
#include "MetaObject/Params/Types.hpp"
#ifdef MO_EXPORTS
#undef MO_EXPORTS
#endif
#if (defined WIN32 || defined _WIN32 || defined WINCE || defined __CYGWIN__) && (defined MetaParams_EXPORTS)
#  define MO_EXPORTS __declspec(dllexport)
#elif defined __GNUC__ && __GNUC__ >= 4
#  define MO_EXPORTS __attribute__ ((visibility ("default")))
#else
#  define MO_EXPORTS
#endif
#include "MetaObject/Params/detail/MetaParamImpl.hpp"

#include <cereal/types/vector.hpp>
#include <cereal/types/string.hpp>

namespace mo{
    namespace IO{
        namespace Text{
            namespace imp{

            template<> bool DeSerialize<EnumParam>(ITParam<EnumParam>* param, std::stringstream& ss){
                ParamTraits<EnumParam>::Storage_t ptr;
                if(param->getData(ptr)){
                    ptr->values.clear();
                    ptr->enumerations.clear();
                    std::string size;
                    std::getline(ss, size, '[');
                    if (size.size()){
                        size_t size_ = boost::lexical_cast<size_t>(size);
                        ptr->values.reserve(size_);
                        ptr->enumerations.reserve(size_);
                    }
                    std::string enumeration;
                    int value;
                    char ch;
                    while( ss >> enumeration >> ch >> value){
                        ptr->addEnum(value, enumeration);
                        ss >> ch;
                    }
                    return true;
                }
                return false;
            }

            template<> bool Serialize<EnumParam>(ITParam<EnumParam>* param, std::stringstream& ss){
                ParamTraits<EnumParam>::Storage_t ptr;
                if(param->getData(ptr)){
                    ss << ptr->enumerations.size();
                    ss << "[";
                    for(int i = 0; i < ptr->enumerations.size(); ++i){
                        if(i != 0)
                            ss << ", ";
                        ss << ptr->enumerations[i] << ":" << ptr->values[i];
                    }
                    ss << "]";
                    return true;
                }
                return false;
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
    template<class Archive> void load(Archive& ar, mo::Writypedefile& m){
        std::string path;
        ar(path);
        m = path;
    }
    template<class Archive> void save(Archive& ar, mo::Writypedefile const& m){
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
INSTANTIATE_META_Param(Writypedefile);
INSTANTIATE_META_Param(ReadDirectory);
INSTANTIATE_META_Param(WriteDirectory);
INSTANTIATE_META_Param(EnumParam);
