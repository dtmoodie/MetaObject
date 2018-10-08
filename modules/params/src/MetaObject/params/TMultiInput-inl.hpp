#ifndef TMULTIINPUTINL_HPP
#define TMULTIINPUTINL_HPP
#include "TMultiInput.hpp"

namespace mo
{
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
        return sizeof...(Ts)-indexOfHelper<T, Ts...>(sizeof...(Ts), static_cast<std::tuple<Ts...>*>(nullptr));
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
    class MO_EXPORTS MultiConnection : public Connection
    {
      public:
        MultiConnection(std::vector<std::shared_ptr<Connection>>&& connections);
        virtual ~MultiConnection() override;
        virtual bool disconnect() override;

      private:
        std::vector<std::shared_ptr<Connection>> m_connections;
    };

    template <class... Types>
    TMultiInput<Types...>::TMultiInput()
        : InputParam()
        , IParam("", mo::ParamFlags::Input_e)
    {
        this->setFlags(mo::ParamFlags::Input_e);
    }

    template <class... Types>
    typename TMultiInput<Types...>::InputTypeTuple TMultiInput<Types...>::initNullptr()
    {
        InputTypeTuple out;
        Initializer init;
        typeLoop<Types...>(init, out);
        return out;
    }

    template <class... Types>
    void TMultiInput<Types...>::setUserDataPtr(std::tuple<const Types*...>* user_var_)
    {
        typeLoop<Types...>(*this, user_var_);
    }

    template <class... Types>
    template <class T>
    void TMultiInput<Types...>::apply(std::tuple<const Types*...>* user_var_)
    {
        mo::get<TInputParamPtr<T>>(m_inputs).setUserDataPtr(&mo::get<const T*>(*user_var_));
    }

} // namespace mo
#endif // TMULTIINPUTINL_HPP
