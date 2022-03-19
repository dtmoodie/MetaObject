#ifndef MO_PYTHON_METAOBJECT_HPP
#define MO_PYTHON_METAOBJECT_HPP

#include <boost/mpl/vector.hpp>
#include <functional>

namespace boost
{
    namespace python
    {
        namespace detail
        {

            template <class T, class... Args, class... OTHER>
            inline boost::mpl::vector<T, Args...> get_signature(const std::function<T(Args...)>&, OTHER&&...)
            {
                return boost::mpl::vector<T, Args...>();
            }

            template <class T, class... Args>
            inline boost::mpl::vector<T, Args...> get_signature(const std::function<T(Args...)>&)
            {
                return boost::mpl::vector<T, Args...>();
            }
        } // namespace detail
    }     // namespace python
} // namespace boost

#include "PythonConversionVisitation.hpp"

#include <MetaObject/object/IMetaObjectInfo.hpp>
#include <MetaObject/params/ParamInfo.hpp>
#include <MetaObject/python/DataConverter.hpp>
#include <MetaObject/python/SlotInvoker.hpp>
#include <MetaObject/python/lambda.hpp>
#include <MetaObject/signals/SlotInfo.hpp>

#include <Python.h>
#include <RuntimeObjectSystem/ObjectInterface.h>
#include <boost/python.hpp>

struct IObjectConstructor;

namespace mo
{
    namespace python
    {
        void MO_EXPORTS setupObjects(std::vector<IObjectConstructor*>& ctrs);
        void MO_EXPORTS setupInterface();
    } // namespace python

    template <class T>
    std::string printObject(const T* obj)
    {
        if (!obj)
        {
            return "NULL";
        }
        auto params = obj->getParams();
        std::stringstream ss;
        for (auto param : params)
        {
            param->print(ss);
            ss << '\n';
        }
        return ss.str();
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
                auto param = obj->getParam(param_names[i]);
                if (param)
                {
                    if (param->checkFlags(mo::ParamFlags::kINPUT))
                    {
                        MO_LOG(warn, "Setting of input parameters not implemented yet");
                        continue;
                    }

                    python::ControlParamSetter setter(args[i]);
                    param->load(setter);
                    if (!setter.success())
                    {
                        MO_LOG(debug, "Unable to set {}", param_names[i]);
                    }
                }
                else
                {
                    MO_LOG(debug, "No parameter named {} ", param_names[i]);
                }
            }
        }
    }

    template <class R, class... Args, int... Is>
    std::function<R(Args...)> ctrBind(R (*p)(IObjectConstructor* ctr, std::vector<std::string>, Args...),
                                      IObjectConstructor* ctr,
                                      std::vector<std::string> param_names,
                                      ct::int_sequence<Is...>)
    {
        return std::bind(p, ctr, param_names, ct::placeholder_template<Is>{}...);
    }

    template <int N, class T, class Storage, class... Args>
    struct CreateMetaObject<N, T, Storage, ct::VariadicTypedef<Args...>>
    {
        static const int size = N;
        using ConstructedType = Storage;

        CreateMetaObject(const std::vector<std::string>& param_names_)
        {
            MO_ASSERT_EQ(param_names_.size(), N);
            for (size_t i = 0; i < param_names_.size(); ++i)
            {
                m_keywords[i] = (boost::python::arg(param_names_[i].c_str()) = boost::python::object());
            }
        }

        static ConstructedType create(IObjectConstructor* ctr, std::vector<std::string> param_names, Args... args)
        {
            rcc::shared_ptr<T> ptr = ctr->Construct();
            ptr->Init(true);
            initializeParameters<T>(ptr, param_names, {args...});
            return ptr;
        }

        static std::function<ConstructedType(Args...)> bind(IObjectConstructor* ctr,
                                                            std::vector<std::string> param_names)
        {
            return ctrBind(&CreateMetaObject<N, T, Storage, ct::VariadicTypedef<Args...>>::create,
                           ctr,
                           param_names,
                           ct::make_int_sequence<N>{});
        }

        boost::python::detail::keyword_range range() const
        {
            return std::make_pair<boost::python::detail::keyword const*, boost::python::detail::keyword const*>(
                &m_keywords[0], &m_keywords[0] + N);
        }

        std::array<boost::python::detail::keyword, N> m_keywords;
    };

    template <int N, class T, class Storage>
    struct CreateMetaObject<N, T, Storage, ct::VariadicTypedef<void>>
    {
        static const int size = N;
        using ConstructedType = Storage;

        CreateMetaObject(const std::vector<std::string>& param_names_)
        {
            MO_ASSERT_EQ(param_names_.size(), N);
            for (size_t i = 0; i < param_names_.size(); ++i)
            {
                m_keywords[i] = (boost::python::arg(param_names_[i].c_str()) = boost::python::object());
            }
        }

        static ConstructedType create(IObjectConstructor* ctr, std::vector<std::string> param_names)
        {
            rcc::shared_ptr<T> ptr = ctr->Construct();
            ptr->Init(true);
            initializeParameters<T>(ptr, param_names, {});
            return ptr;
        }

        static std::function<ConstructedType()> bind(IObjectConstructor* ctr, std::vector<std::string> param_names)
        {
            return [ctr, param_names]() { return create(ctr, param_names); };
        }

        boost::python::detail::keyword_range range() const
        {
            return std::make_pair<boost::python::detail::keyword const*, boost::python::detail::keyword const*>(
                &m_keywords[0], &m_keywords[0] + N);
        }

        std::array<boost::python::detail::keyword, N> m_keywords;
    };

    template <class T,
              int N,
              class Storage,
              template <int N1, class T1, class S, class... Args> class Creator = CreateMetaObject>
    boost::python::object makeConstructorHelper(IObjectConstructor* ctr, const std::vector<std::string>& param_names)
    {
        using Signature_t = typename FunctionSignatureBuilder<boost::python::object, N - 1>::VariadicTypedef_t;
        using Creator_t = Creator<N, T, Storage, Signature_t>;
        Creator_t creator(param_names);
        return boost::python::make_constructor(
            Creator_t::bind(ctr, param_names), boost::python::default_call_policies(), creator);
    }

    template <class T>
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

    bool setParam(mo::IControlParam* param, const boost::python::object& python_obj);

    template <class T>
    bool setParamHelper(std::string name, T& obj, const boost::python::object& python_obj)
    {
        mo::IControlParam* param = obj.getParam(name);
        if (param)
        {
            return setParam(param, python_obj);
        }
        return false;
    }

    boost::python::object getParam(const mo::IControlParam* param);
    boost::python::object getParam(mo::IPublisher* param);

    template <class T>
    boost::python::object getParamHelper(std::string name, T& obj)
    {
        const mo::IControlParam* param = obj.getParam(name);
        if (param)
        {
            return getParam(param);
        }
        mo::IPublisher* pub = obj.getOutput(name);
        if (pub)
        {
            return getParam(pub);
        }
        return {};
    }

    template <class T, class BP>
    void addParamAccessors(BP& bpobj, const IMetaObjectInfo* minfo)
    {
        std::vector<ParamInfo*> param_infos = minfo->getParamInfo();
        for (auto param_info : param_infos)
        {
            std::function<boost::python::object(T&)> getter_func(
                std::bind(getParamHelper<T>, param_info->getName(), std::placeholders::_1));

            std::function<bool(T&, const boost::python::object&)> setter_func(
                std::bind(setParamHelper<T>, param_info->getName(), std::placeholders::_1, std::placeholders::_2));

            bpobj.add_property(param_info->getName().c_str(), getter_func, setter_func);
        }
    }

    template <class T>
    mo::IPublisher* getPublisher(std::string name, T& obj)
    {
        return obj.getOutput(name);
    }

    template <class T, class BP>
    void addOutputAccessors(BP& bpobj, const IMetaObjectInfo* minfo)
    {
        std::vector<ParamInfo*> param_infos = minfo->getParamInfo();
        for (ParamInfo* param_info : param_infos)
        {
            const std::string& name = param_info->getName();
            ParamFlags flags = param_info->getParamFlags();
            if (flags == ParamFlags::kOUTPUT)
            {
                std::function<mo::IPublisher*(T&)> output_getter(
                    std::bind(&getPublisher<T>, name, std::placeholders::_1));
                bpobj.add_property(
                    param_info->getName().c_str(),
                    boost::python::make_getter(output_getter, boost::python::return_internal_reference<>()));
            }
        }
    }

    template <class R, class T, class... Args, int... Is>
    std::function<R(T&, const Args&...)>
    slotBind(R (*p)(const std::string&, T&, const Args&...), const std::string& slot_name, ct::int_sequence<Is...>)
    {
        return std::bind(p, slot_name, ct::placeholder_template<Is>{}...);
    }

    template <class R, class T, int... Is>
    std::function<R(T&)> slotBind(R (*p)(const std::string&, T&), const std::string& slot_name, ct::int_sequence<1>)
    {
        return std::bind(p, slot_name, std::placeholders::_1);
    }

    template <class R, class... Args, int... Is>
    std::function<R(Args...)> staticSlotBind(R (*p)(mo::TSlot<R(Args...)>*, const Args&...),
                                             mo::TSlot<R(Args...)>* slot_ptr,
                                             ct::int_sequence<Is...>)
    {
        return std::bind(p, slot_ptr, ct::placeholder_template<Is>{}...);
    }

    template <class T, class R, class... Args, class BP>
    void addSlotAccessors(BP& bpobj, const IMetaObjectInfo* minfo)
    {
        std::vector<SlotInfo*> slot_infos = minfo->getSlotInfo();
        for (SlotInfo* slot_info : slot_infos)
        {
            if (slot_info->signature == TypeInfo::create<R(Args...)>())
            {
                bpobj.def(slot_info->name.c_str(),
                          std::function<R(T&, const Args&...)>(slotBind(&mo::python::SlotInvoker<T, R(Args...)>::invoke,
                                                                        slot_info->name,
                                                                        ct::make_int_sequence<sizeof...(Args) + 1>{})));
                if (slot_info->is_static)
                {
                    bpobj.staticmethod(slot_info->name.c_str());
                }
            }
        }
    }

    template <class Sig>
    struct StaticSlotAccessor;

    template <class R, class... Args>
    struct StaticSlotAccessor<R(Args...)>
    {
        template <class T, class BP>
        static void add(BP& bpobj, const IMetaObjectInfo* minfo)
        {
            auto static_slots = minfo->getStaticSlots();
            for (const auto& slot : static_slots)
            {
                if (slot.first->getSignature() == TypeInfo::create<R(Args...)>())
                {
                    auto tslot = dynamic_cast<TSlot<R(Args...)>*>(slot.first);
                    bpobj.def(slot.second.c_str(),
                              std::function<R(const Args&...)>(
                                  staticSlotBind(&mo::python::StaticSlotInvoker<R(Args...)>::invoke,
                                                 tslot,
                                                 ct::make_int_sequence<sizeof...(Args)>{})));
                    bpobj.staticmethod(slot.second.c_str());
                }
            }
        }
    };

    struct DefaultParamPolicy
    {
        std::vector<std::string> operator()(const std::vector<ParamInfo*>& param_info)
        {
            std::vector<std::string> param_names;
            for (const ParamInfo* pinfo : param_info)
            {
                param_names.push_back(pinfo->getName());
            }
            return param_names;
        }
    };

    template <class T,
              class Storage = rcc::shared_ptr<T>,
              template <int N, class T1, class S, class... Args> class Creator = CreateMetaObject,
              class ParamPolicy = DefaultParamPolicy>
    boost::python::object makeConstructor(IObjectConstructor* ctr, ParamPolicy policy = ParamPolicy())
    {
        auto minfo = dynamic_cast<const IMetaObjectInfo*>(ctr->GetObjectInfo());
        if (!minfo)
        {
            return {};
        }
        std::vector<ParamInfo*> param_info = minfo->getParamInfo();
        auto param_names = policy(param_info);
        switch (param_names.size())
        {
        case 0:
        {
            return makeConstructorHelper<T, 0, Storage, Creator>(ctr, param_names);
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
} // namespace mo

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
        } // namespace detail
    }     // namespace python
} // namespace boost
#endif // MO_PYTHON_METAOBJECT_HPP
