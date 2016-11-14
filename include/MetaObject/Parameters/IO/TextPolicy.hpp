#pragma once
#include <MetaObject/Parameters/IParameter.hpp>
#include <MetaObject/Parameters/MetaParameter.hpp>
#include "SerializationFunctionRegistry.hpp"
#include <boost/lexical_cast.hpp>

namespace mo 
{ 
namespace IO 
{ 
namespace Text 
{
    namespace imp
    {
        template<typename T>
        auto Serialize_imp(std::ostream& os, T const& obj, int) ->decltype(os << obj, void())
        {
            os << obj;
        }

        template<typename T>
        void Serialize_imp(std::ostream& os, T const& obj, long)
        {

        }

        template<typename T>
        auto DeSerialize_imp(std::istream& is, T& obj, int) ->decltype(is >> obj, void())
        {
            is >> obj;
        }
        template<typename T>
        void DeSerialize_imp(std::istream& is, T& obj, long)
        {

        }

        template<typename T>
        bool Serialize(ITypedParameter<T>* param, std::stringstream& ss)
        {
            T* ptr = param->GetDataPtr();
            if (ptr)
            {
                Serialize_imp(ss, *ptr, 0);
                //ss << *ptr;
                return true;
            }
            return false;
        }

        template<typename T>
        bool DeSerialize(ITypedParameter<T>* param, std::stringstream& ss)
        {
            T* ptr = param->GetDataPtr();
            if (ptr)
            {
                //ss >> *ptr;
                DeSerialize_imp(ss, *ptr, 0);
                return true;
            }
            return false;
        }
        template<typename T> bool Serialize(ITypedParameter<std::vector<T>>* param, std::stringstream& ss)
        {
            std::vector<T>* ptr = param->GetDataPtr();
            if (ptr)
            {
                ss << ptr->size();
                ss << "[";
                for (size_t i = 0; i < ptr->size(); ++i)
                {
                    if (i != 0)
                        ss << ", ";
                    ss << (*ptr)[i];
                }
                ss << "]";
                return true;
            }
            return false;
        }
        template<typename T> bool DeSerialize(ITypedParameter<std::vector<T>>* param, std::stringstream& ss)
        {
            std::vector<T>* ptr = param->GetDataPtr();
            if (ptr)
            {
                ptr->clear();
                std::string size;
                std::getline(ss, size, '[');
                if (size.size())
                {
                    ptr->reserve(boost::lexical_cast<size_t>(size));
                }
                T value;
                char ch; // For flushing the ','
                while (ss >> value)
                {
                    ss >> ch;
                    ptr->push_back(value);
                }
                return true;
            }
            return false;
        }
    }
    

    template<typename T> bool WrapSerialize(IParameter* param, std::stringstream& ss)
    {
        ITypedParameter<T>* typed = dynamic_cast<ITypedParameter<T>*>(param);
        if (typed)
        {
            if(imp::Serialize(typed, ss))
            {
                return true;
            }
        }
        return false;
    }

    template<typename T> bool WrapDeSerialize(IParameter* param, std::stringstream& ss)
    {
        ITypedParameter<T>* typed = dynamic_cast<ITypedParameter<T>*>(param);
        if (typed)
        {
            if(imp::DeSerialize(typed, ss))
            {
                typed->Commit();
                return true;
            }
        }
        return false;
    }
    
    template<class T> struct Policy
    {
        Policy()
        {
            SerializationFunctionRegistry::Instance()->SetTextSerializationFunctions(
                TypeInfo(typeid(T)),
                std::bind(&WrapSerialize<T>, std::placeholders::_1, std::placeholders::_2),
                std::bind(&WrapDeSerialize<T>, std::placeholders::_1, std::placeholders::_2));
        }
    };
} // Text 
} // IO

#define PARAMETER_TEXT_SERIALIZATION_POLICY_INST_(N) \
  template<class T> struct MetaParameter<T, N, void>: public MetaParameter<T, N - 1, void>, public IO::Text::Policy<T> \
    { \
        MetaParameter(const char* name): \
            MetaParameter<T, N-1, void>(name){} \
    };

PARAMETER_TEXT_SERIALIZATION_POLICY_INST_(__COUNTER__)
} // mo
