#ifndef MO_PARAMS_TMULTISUBSCRIBER_HPP
#define MO_PARAMS_TMULTISUBSCRIBER_HPP
#include "IMultiSubscriber.hpp"

#include "TSubscriberPtr.hpp"
#include "TypeSelector.hpp"

#include <ct/Indexer.hpp>

#include <tuple>
namespace mo
{
    template <class... Types>
    class TMultiSubscriber : virtual public IMultiSubscriber
    {
      public:
        using InputTypeTuple = std::tuple<const Types*...>;
        using TypeTuple = std::tuple<Types...>;
        static InputTypeTuple initNullptr();

        TMultiSubscriber();

        void setUserDataPtr(std::tuple<const Types*...>* user_var_);

        template <class T>
        inline void apply(std::tuple<const Types*...>* user_var_);
        std::vector<TypeInfo> getInputTypes() const
        {
            return {TypeInfo::create<Types>()...};
        }

      private:
        void onInputUpdate(const IDataContainerPtr_t&, IParam*, UpdateFlags);
        std::tuple<TSubscriberPtr<Types>...> m_inputs;
    };

    /////////////////////////////////////////////////////////////////////////////////////
    // Implementation
    /////////////////////////////////////////////////////////////////////////////////////
    template <class T, class U>
    constexpr int indexOfHelper(int idx, std::tuple<U>* = nullptr)
    {
        return idx;
    }
    template <class T, class U, class... Ts>
    constexpr int indexOfHelper(int idx, std::tuple<U, Ts...>* = nullptr)
    {
        return std::is_same<T, U>::value ? idx
                                         : indexOfHelper<T, Ts...>(idx - 1, static_cast<std::tuple<Ts...>*>(nullptr));
    }

    template <class T, class... Ts>
    constexpr int indexOf()
    {
        return sizeof...(Ts) - indexOfHelper<T, Ts...>(sizeof...(Ts), static_cast<std::tuple<Ts...>*>(nullptr));
    }

    template <class T, class... Ts>
    inline T& get(std::tuple<Ts...>& tuple)
    {
        return std::get<indexOf<T, Ts...>()>(tuple);
    }

    template <class T, class... Ts>
    inline const T& get(const std::tuple<Ts...>& tuple)
    {
        return std::get<indexOf<T, Ts...>()>(tuple);
    }

    template <class T>
    struct AcceptInputRedirect
    {
        AcceptInputRedirect(const T& func)
            : m_func(func)
        {
        }
        template <class Type, class... Args>
        void apply(Args&&... args)
        {
            m_func.template acceptsInput<Type>(std::forward<Args>(args)...);
        }
        const T& m_func;
    };

    struct Initializer
    {
        template <class Type, class... Args>
        void apply(std::tuple<const Args*...>& tuple)
        {
            mo::get<const Type*>(tuple) = nullptr;
        }
    };

    template <class T, class... Args>
    void globHelper(std::vector<T*>& vec, std::tuple<Args...>& tuple, ct::Indexer<0>)
    {
        vec.push_back(&std::get<0>(tuple));
    }

    template <class T, class... Args, ct::index_t I>
    void globHelper(std::vector<T*>& vec, std::tuple<Args...>& tuple, ct::Indexer<I> idx)
    {
        vec.push_back(&std::get<I>(tuple));
        globHelper(vec, tuple, --idx);
    }

    template <class T, class... Args>
    std::vector<T*> globParamPtrs(std::tuple<Args...>& tuple)
    {
        std::vector<T*> out;
        globHelper(out, tuple, ct::Indexer<sizeof...(Args) - 1>{});
        return out;
    }

    template <class... Types>
    TMultiSubscriber<Types...>::TMultiSubscriber()
        : m_inputs()
        , IMultiSubscriber()
    {
        IMultiSubscriber::setInputs(globParamPtrs<ISubscriber>(m_inputs));
        this->setFlags(mo::ParamFlags::kINPUT);
    }

    template <class... Types>
    typename TMultiSubscriber<Types...>::InputTypeTuple TMultiSubscriber<Types...>::initNullptr()
    {
        InputTypeTuple out;
        Initializer init;
        typeLoop<Types...>(init, out);
        return out;
    }

    template <class... Types>
    void TMultiSubscriber<Types...>::setUserDataPtr(std::tuple<const Types*...>* user_var_)
    {
        typeLoop<Types...>(*this, user_var_);
    }

    template <class... Types>
    template <class T>
    void TMultiSubscriber<Types...>::apply(std::tuple<const Types*...>* user_var_)
    {
        mo::get<TSubscriberPtr<T>>(m_inputs).setUserDataPtr(&mo::get<const T*>(*user_var_));
    }

    template <class... Types>
    void TMultiSubscriber<Types...>::onInputUpdate(const IDataContainerPtr_t&, IParam*, UpdateFlags)
    {
    }
} // namespace mo

#define MULTI_INPUT(name, ...)                                                                                         \
    mo::TMultiSubscriber<__VA_ARGS__> name##_param;                                                                    \
    typename mo::TMultiSubscriber<__VA_ARGS__>::InputTypeTuple name;                                                   \
                                                                                                                       \
  public:                                                                                                              \
    constexpr static ct::MemberObjectPointer<mo::TMultiSubscriber<__VA_ARGS__> DataType::*,                            \
                                             ct::Flags::READABLE | ct::Flags::WRITABLE>                                \
    getPtr(const ct::Indexer<__COUNTER__ - REFLECT_COUNT_BEGIN>)                                                       \
    {                                                                                                                  \
        return ct::makeMemberObjectPointer(#name, &DataType::name##_param);                                            \
    }

#endif // MO_PARAMS_TMULTISUBSCRIBER_HPP