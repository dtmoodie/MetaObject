#pragma once
#include "TInputParam.hpp"
#include "TypeSelector.hpp"

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
        AcceptInputRedirect(const T& func) : m_func(func) {}
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

    template <class... Types>
    class TMultiInput : virtual public InputParam
    {
      public:
        using InputTypeTuple = std::tuple<const Types*...>;
        using TypeTuple = std::tuple<Types...>;
        static InputTypeTuple initNullptr();

        TMultiInput();

        void setUserDataPtr(std::tuple<const Types*...>* user_var_);

        bool setInput(std::shared_ptr<IParam> input) override;

        bool setInput(IParam* input) override;

        bool getInput(const OptionalTime_t& ts, size_t* fn = nullptr) override;

        bool getInput(size_t fn, OptionalTime_t* ts = nullptr) override;

        void setMtx(Mutex_t* mtx) override;

        mo::TypeInfo getTypeInfo() const override;

        mo::IParam* getInputParam() const override;

        OptionalTime_t getInputTimestamp() override;

        size_t getInputFrameNumber() override;

        bool isInputSet() const override;

        bool acceptsInput(IParam* input) const override;

        bool acceptsType(const TypeInfo& type) const override;

        template <class T>
        inline void acceptsInput(IParam* input, bool* success) const;

        template <class T>
        inline void acceptsInput(const TypeInfo& type, bool* success) const;

        template <class T>
        inline void apply(std::tuple<const Types*...>* user_var_);
        template <class T>
        inline void apply(Mutex_t* mtx);
        template <class T>
        inline void apply(Context* ctx);
        template <class T>
        inline void apply(const OptionalTime_t& ts, size_t* fn, bool* success);
        template <class T>
        inline void apply(size_t fn, OptionalTime_t* ts, bool* success);
        template <class T>
        inline void apply(std::shared_ptr<IParam> input, bool* success);
        template <class T>
        inline void apply(IParam* input, bool* success);

      private:
        static mo::TypeInfo _void_type_info;
        std::tuple<TInputParamPtr<Types>...> m_inputs;
        mo::IParam* m_current_input = nullptr;
    };

    template <class... T>
    mo::TypeInfo TMultiInput<T...>::_void_type_info = mo::TypeInfo(typeid(void));
}

#include "TMultiInput-inl.hpp"
