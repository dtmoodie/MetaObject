#pragma once
#include "TInputParam.hpp"
#include "TypeSelector.hpp"

namespace mo
{
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

        virtual OptionalTime_t getTimestamp() const override;
        virtual size_t getFrameNumber() const;

        size_t getInputFrameNumber() override;

        bool isInputSet() const override;

        bool acceptsInput(IParam* input) const override;

        bool acceptsType(const TypeInfo& type) const override;

        template <class T>
        inline void acceptsInput(IParam* input, bool* success) const;

        template <class T>
        inline void acceptsInput(const TypeInfo& type, bool* success) const;

        template <class T>
        inline void apply(OptionalTime_t* ts) const;

        template <class T>
        inline void apply(size_t* fn) const;

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

        template <class T, class Slot>
        inline void apply(std::vector<ConnectionPtr_t>& connection, Slot slot);

        // Virtual to allow typed overload for interface slot input
        virtual ConnectionPtr_t registerUpdateNotifier(ISlot* f) override;
        virtual ConnectionPtr_t registerUpdateNotifier(const ISignalRelay::Ptr& relay) override;

        template <class T>
        inline void apply(bool* modified) const;
        template <class T>
        inline void apply(const bool modified);
        virtual bool modified() const override;
        virtual void modified(bool value) override;

      private:
        static mo::TypeInfo _void_type_info;
        std::tuple<TInputParamPtr<Types>...> m_inputs;
        mo::IParam* m_current_input = nullptr;
    };

    template <class... T>
    mo::TypeInfo TMultiInput<T...>::_void_type_info = mo::TypeInfo(typeid(void));
}

#define MULTI_INPUT(name, ...)                                                                                         \
    mo::TMultiInput<__VA_ARGS__> name##_param;                                                                         \
    typename mo::TMultiInput<__VA_ARGS__>::InputTypeTuple name;                                                        \
    VISIT(name, mo::INPUT, mo::TMultiInput<__VA_ARGS__>::initNullptr())
