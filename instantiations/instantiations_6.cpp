#include "MetaObject/Parameters/MetaParameter.hpp"
#include "MetaObject/Parameters/UI/Qt/OpenCV.hpp"
#include "MetaObject/Parameters/UI/Qt/Containers.hpp"
#include "MetaObject/Parameters/UI/Qt/TParameterProxy.hpp"
#include "MetaObject/Parameters/Buffers/CircularBuffer.hpp"
#include "MetaObject/Parameters/Buffers/StreamBuffer.hpp"
#include "MetaObject/Parameters/Buffers/map.hpp"
#include "MetaObject/Parameters/IO/CerealPolicy.hpp"
#include "MetaObject/Parameters/IO/TextPolicy.hpp"
#include "instantiate.hpp"
#include "MetaObject/Parameters/Types.hpp"
#include <cereal/types/vector.hpp>
#include <cereal/types/string.hpp>


namespace mo
{
    namespace IO
    {
        namespace Text
        {
            namespace imp
            {
            
            template<> bool DeSerialize<EnumParameter>(ITypedParameter<EnumParameter>* param, std::stringstream& ss)
            {
                EnumParameter* ptr = param->GetDataPtr();
                if(ptr)
                {
                    ptr->values.clear();
                    ptr->enumerations.clear();
                    std::string size;
                    std::getline(ss, size, '[');
                    if (size.size())
                    {
                        size_t size_ = boost::lexical_cast<size_t>(size);
                        ptr->values.reserve(size_);
                        ptr->enumerations.reserve(size_);
                    }
                    std::string enumeration;
                    int value;
                    char ch;
                    while( ss >> enumeration >> ch >> value)
                    {
                        ptr->addEnum(value, enumeration);
                        ss >> ch;
                    }
                    return true;
                }
                return false;
            }

            template<> bool Serialize<EnumParameter>(ITypedParameter<EnumParameter>* param, std::stringstream& ss)
            {
                EnumParameter* ptr = param->GetDataPtr();
                if (ptr)
                {
                    ss << ptr->enumerations.size();
                    ss << "[";
                    for(int i = 0; i < ptr->enumerations.size(); ++i)
                    {
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

namespace cereal
{
    template<class Archive> void load(Archive& ar, mo::ReadFile& m)
    {
        std::string path;
        ar(path);
        m = path;
    }
    template<class Archive> void save(Archive& ar, mo::ReadFile const & m)
    {
        std::string path = m.string();
        ar(path);
    }
    template<class Archive> void serialize(Archive& ar, mo::WriteFile& m)
    {
        ar(m);
    }
    template<class Archive> void serialize(Archive& ar, mo::ReadDirectory& m)
    {
        ar(m);
    }
    template<class Archive> void serialize(Archive& ar, mo::WriteDirectory& m)
    {
        ar(m);
    }
    template<class Archive> void save(Archive& ar, mo::EnumParameter const& m)
    {
        ar(m.enumerations, m.values);
    }
    template<class Archive> void load(Archive& ar, mo::EnumParameter& m)
    {
        ar(m.enumerations, m.values);
    }
}

using namespace mo;
INSTANTIATE_META_PARAMETER(ReadFile);
INSTANTIATE_META_PARAMETER(WriteFile);
INSTANTIATE_META_PARAMETER(ReadDirectory);
INSTANTIATE_META_PARAMETER(WriteDirectory);
INSTANTIATE_META_PARAMETER(EnumParameter);


