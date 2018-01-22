#pragma once
#include <MetaObject/python/DataConverter.hpp>
#include <RuntimeObjectSystem/ObjectInterface.h>
#include <ce/VariadicTypedef.hpp>
#include <boost/python.hpp>

template<class T, int N>
struct FunctionSignatureBuilder
{
    typedef typename ce::append_to_tupple<T, typename FunctionSignatureBuilder<T, N - 1>::VariadicTypedef_t>::type VariadicTypedef_t;
};

template<class T>
struct FunctionSignatureBuilder<T, 0>
{
    typedef ce::variadic_typedef<T> VariadicTypedef_t;
};

struct IObjectConstructor;

namespace mo
{
    template<int N, class Type, class ... Args>
    struct CreateMetaObject{};

    template<class T>
    void initializeParameters(rcc::shared_ptr<T>& obj, const std::vector<std::string>& param_names, const std::vector<boost::python::object>& args)
    {
        MO_ASSERT(param_names.size() == args.size());
        for (size_t i = 0; i < param_names.size(); ++i)
        {
            if (args[i])
            {
                IParam* param = obj->getParamOptional(param_names[i]);
                if (param)
                {
                    if (param->checkFlags(mo::ParamFlags::Input_e))
                    {
                        MO_LOG(warning) << "Setting of input parameters not implemented yet";
                        continue;
                    }
                    auto setter = python::DataConverterRegistry::instance()->getSetter(param->getTypeInfo());
                    if (setter)
                    {
                        if (!setter(param, args[i]))
                        {
                            MO_LOG(debug) << "Unable to set " << param_names[i];
                        }
                    }
                    else
                    {
                        MO_LOG(debug) << "No converter available for " << mo::Demangle::typeToName(param->getTypeInfo());
                    }
                }
                else
                {
                    MO_LOG(debug) << "No parameter named " << param_names[i];
                }
            }
        }
    }

    template<int N, class T, class ... Args>
    struct CreateMetaObject<N, T, ce::variadic_typedef<Args...>>
    {
        static const int size = N;
        
        CreateMetaObject(const std::vector<std::string>& param_names_)
        {
            MO_CHECK_EQ(param_names_.size(), N);
            for (size_t i = 0; i < param_names_.size(); ++i)
            {
                m_keywords[i] = (boost::python::arg(param_names_[i].c_str()) = boost::python::object());
            }
        }

        static rcc::shared_ptr<T> create(IObjectConstructor* ctr, std::vector<std::string> param_names, Args... args)
        {
            IObject* obj = ctr->Construct();
            rcc::shared_ptr<T> ptr(obj);
            ptr->Init(true);
            initializeParameters<T>(ptr, param_names, { args... });
            return obj;
        }

        boost::python::detail::keyword_range range() const
        {
            return  std::make_pair<boost::python::detail::keyword const*, boost::python::detail::keyword const*>(&m_keywords[0], &m_keywords[0] + N);
        }
        
        std::array<boost::python::detail::keyword, N> m_keywords;
    };
    
    template<class R, class ... Args, int... Is>
    std::function<R(Args...)> ctrBind(R (*p)(IObjectConstructor* ctr, std::vector<std::string>, Args...), 
        IObjectConstructor* ctr, std::vector<std::string> param_names, int_sequence<Is...>)
    {
        return std::bind(p, ctr, param_names, placeholder_template<Is>{}...);
    }

    template< class T, int N >
    boost::python::object makeConstructorHelper(IObjectConstructor* ctr, const std::vector<std::string>& param_names)
    {
        typedef FunctionSignatureBuilder<boost::python::object, N - 1>::VariadicTypedef_t Signature_t;
        typedef CreateMetaObject<N, T, Signature_t> Creator_t;
        Creator_t creator(param_names);
        return boost::python::make_constructor(
            ctrBind<rcc::shared_ptr<T>>(Creator_t::create, ctr, param_names, make_int_sequence<N>{}),
            boost::python::default_call_policies(),
            creator);
    }

    template<class T>
    boost::python::object makeConstructor(IObjectConstructor* ctr)
    {
        IMetaObjectInfo* minfo = dynamic_cast<IMetaObjectInfo*>(ctr->GetObjectInfo());
        std::vector<ParamInfo*> param_info = minfo->getParamInfo();
        std::vector<std::string> param_names;
        for (ParamInfo* pinfo : param_info)
        {
            param_names.push_back(pinfo->name);
        }
        switch (param_names.size())
        {
        case 0:
        {
            break;
        }
        case 1:  {return makeConstructorHelper<T, 1 >(ctr, param_names);}
        case 2:  {return makeConstructorHelper<T, 2 >(ctr, param_names); }
        case 3:  {return makeConstructorHelper<T, 3 >(ctr, param_names); }
        case 4:  {return makeConstructorHelper<T, 4 >(ctr, param_names); }
        case 5:  {return makeConstructorHelper<T, 5 >(ctr, param_names); }
        case 6:  {return makeConstructorHelper<T, 6 >(ctr, param_names); }
        case 7:  {return makeConstructorHelper<T, 7 >(ctr, param_names); }
        case 8:  {return makeConstructorHelper<T, 8 >(ctr, param_names); }
        case 9:  {return makeConstructorHelper<T, 9 >(ctr, param_names); }
        case 10: {return makeConstructorHelper<T, 10>(ctr, param_names); }
        case 11: {return makeConstructorHelper<T, 11>(ctr, param_names); }
        case 12: {return makeConstructorHelper<T, 12>(ctr, param_names); }
        case 13: {return makeConstructorHelper<T, 13>(ctr, param_names); }
        }
        return boost::python::object();
    }

}

namespace boost
{
    namespace python
    {
        namespace detail
        {
            template<int N, class ... T>
            struct is_keywords<mo::CreateMetaObject<N, T...> >
            {
                static const bool value = true;
            };
        }
    }
}