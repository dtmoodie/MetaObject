#pragma once

#include <boost/mpl/vector.hpp>
#include <functional>

namespace boost
{
    namespace python
    {
        namespace detail
        {

            template <class T, class... Args>
            inline boost::mpl::vector<T, Args...> get_signature(std::function<T(Args...)>, void* = 0)
            {
                return boost::mpl::vector<T, Args...>();
            }
        }
    }
}


#include <MetaObject/object/IMetaObjectInfo.hpp>
#include <MetaObject/params/ParamInfo.hpp>
#include <MetaObject/python/DataConverter.hpp>

#include <RuntimeObjectSystem/ObjectInterface.h>
#include <boost/python.hpp>

struct IObjectConstructor;

namespace mo
{
    namespace python
    {
        void MO_EXPORTS setupObjects(std::vector<IObjectConstructor*>& ctrs);
        void MO_EXPORTS setupInterface();
    }


    template <int N, class Type, class Storage, class... Args>
    struct CreateMetaObject
    {
    };

    template <class T>
    void initializeParameters(rcc::shared_ptr<T>& obj,
                              const std::vector<std::string>& param_names,
                              const std::vector<boost::python::object>& args)
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
                        MO_LOG(debug) << "No converter available for "
                                      << mo::Demangle::typeToName(param->getTypeInfo());
                    }
                }
                else
                {
                    MO_LOG(debug) << "No parameter named " << param_names[i];
                }
            }
        }
    }

    template <int N, class T, class Storage, class... Args>
    struct CreateMetaObject<N, T, Storage, ce::variadic_typedef<Args...>>
    {
        static const int size = N;
        typedef Storage ConstructedType;

        CreateMetaObject(const std::vector<std::string>& param_names_)
        {
            MO_CHECK_EQ(param_names_.size(), N);
            for (size_t i = 0; i < param_names_.size(); ++i)
            {
                m_keywords[i] = (boost::python::arg(param_names_[i].c_str()) = boost::python::object());
            }
        }

        static ConstructedType create(IObjectConstructor* ctr, std::vector<std::string> param_names, Args... args)
        {
            IObject* obj = ctr->Construct();
            rcc::shared_ptr<T> ptr(obj);
            ptr->Init(true);
            initializeParameters<T>(ptr, param_names, {args...});
            return ptr;
        }

        boost::python::detail::keyword_range range() const
        {
            return std::make_pair<boost::python::detail::keyword const*, boost::python::detail::keyword const*>(
                &m_keywords[0], &m_keywords[0] + N);
        }

        std::array<boost::python::detail::keyword, N> m_keywords;
    };

    template <class R, class... Args, int... Is>
    std::function<R(Args...)> ctrBind(R (*p)(IObjectConstructor* ctr, std::vector<std::string>, Args...),
                                      IObjectConstructor* ctr,
                                      std::vector<std::string> param_names,
                                      int_sequence<Is...>)
    {
        return std::bind(p, ctr, param_names, placeholder_template<Is>{}...);
    }

    template <class T, int N, class Storage, template <int N, class T, class S, class... Args> class Creator = CreateMetaObject>
    boost::python::object makeConstructorHelper(IObjectConstructor* ctr, const std::vector<std::string>& param_names)
    {
        typedef typename FunctionSignatureBuilder<boost::python::object, N - 1>::VariadicTypedef_t Signature_t;
        typedef Creator<N, T, Storage, Signature_t> Creator_t;
        typedef typename Creator_t::ConstructedType ReturnType;
        Creator_t creator(param_names);
        return boost::python::make_constructor(
            ctrBind<ReturnType>(Creator_t::create, ctr, param_names, make_int_sequence<N>{}),
            boost::python::default_call_policies(),
            creator);
    }

    template<class T>
    rcc::shared_ptr<T> constructObject(IObjectConstructor* ctr)
    {
        rcc::shared_ptr<T> output;
        auto obj = ctr->Construct();
        if (obj)
        {
            output = obj;
            output->Init(true);
        }
        return output;
    }

    template<class T>
    bool setParamHelper(python::DataConverterRegistry::Set_t setter, std::string name,
                        T& obj, const boost::python::object& python_obj)
    {
        auto param = obj.getParamOptional(name);
        if (param)
        {
            return setter(param, python_obj);
        }
        return false;
    }

    template<class T>
    boost::python::object getParamHelper(python::DataConverterRegistry::Get_t getter, std::string name,
        const T& obj)
    {
        auto param = obj.getParamOptional(name);
        if (param)
        {
            return getter(param);
        }
        return {};
    }

    template<class T, class BP>
    void addParamAccessors(BP& bpobj, IMetaObjectInfo* minfo)
    {
        std::vector<ParamInfo*> param_infos = minfo->getParamInfo();
        for (auto param_info : param_infos)
        {
            auto setter = python::DataConverterRegistry::instance()->getSetter(param_info->data_type);
            auto getter = python::DataConverterRegistry::instance()->getGetter(param_info->data_type);
            if (setter && getter)
            {
                bpobj.def(("get_" + param_info->name).c_str(), std::function<boost::python::object(const T&)>(std::bind(getParamHelper<T>, getter, param_info->name, std::placeholders::_1)));
                bpobj.def(("set_" + param_info->name).c_str(), std::function<bool(T&, const boost::python::object&)>(std::bind(setParamHelper<T>, setter, param_info->name, std::placeholders::_1, std::placeholders::_2)));
            }
        }
    }

    template <class T, class Storage = rcc::shared_ptr<T>, template <int N, class T, class Storage, class... Args> class Creator = CreateMetaObject>
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
        case 1:
        {
            return makeConstructorHelper<T, 1, Storage, Creator>(ctr, param_names);
        }
        case 2:
        {
            return makeConstructorHelper<T, 2, Storage, Creator>(ctr, param_names);
        }
        case 3:
        {
            return makeConstructorHelper<T, 3, Storage, Creator>(ctr, param_names);
        }
        case 4:
        {
            return makeConstructorHelper<T, 4, Storage, Creator>(ctr, param_names);
        }
        case 5:
        {
            return makeConstructorHelper<T, 5, Storage, Creator>(ctr, param_names);
        }
        case 6:
        {
            return makeConstructorHelper<T, 6, Storage, Creator>(ctr, param_names);
        }
        case 7:
        {
            return makeConstructorHelper<T, 7, Storage, Creator>(ctr, param_names);
        }
        case 8:
        {
            return makeConstructorHelper<T, 8, Storage, Creator>(ctr, param_names);
        }
        case 9:
        {
            return makeConstructorHelper<T, 9, Storage, Creator>(ctr, param_names);
        }
        case 10:
        {
            return makeConstructorHelper<T, 10, Storage, Creator>(ctr, param_names);
        }
        case 11:
        {
            return makeConstructorHelper<T, 11, Storage, Creator>(ctr, param_names);
        }
        case 12:
        {
            return makeConstructorHelper<T, 12, Storage, Creator>(ctr, param_names);
        }
        case 13:
        {
            return makeConstructorHelper<T, 13, Storage, Creator>(ctr, param_names);
        }
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
            template <int N, class... T>
            struct is_keywords<mo::CreateMetaObject<N, T...>>
            {
                static const bool value = true;
            };
        }
    }
}
