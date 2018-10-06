#pragma once
#include "TInputParam.hpp"
#include "TypeSelector.hpp"

namespace mo
{
    class IMultiInput : public InputParam
    {
      public:
        IMultiInput(const std::vector<InputParam*>& inputs);

        virtual bool setInput(const std::shared_ptr<IParam>& input) override;
        virtual bool setInput(IParam* input) override;
        virtual bool getInputData(const Header& desired, Header* retrieved) override;
        virtual void setMtx(Mutex_t* mtx) override;

        mo::TypeInfo getTypeInfo() const override;
        mo::IParam* getInputParam() const override;

        OptionalTime_t getInputTimestamp() override;
        virtual uint64_t getInputFrameNumber() override;

        virtual OptionalTime_t getTimestamp() const override;
        virtual uint64_t getFrameNumber() const override;

        bool isInputSet() const override;

        bool acceptsInput(IParam* input) const override;

        bool acceptsType(const TypeInfo& type) const override;

        // Virtual to allow typed overload for interface slot input
        virtual ConnectionPtr_t registerUpdateNotifier(ISlot* f) override;
        virtual ConnectionPtr_t registerUpdateNotifier(const ISignalRelay::Ptr& relay) override;

        virtual bool modified() const override;
        virtual void modified(bool value) override;

      private:
        std::vector<InputParam*> m_inputs;
        InputParam* m_current_input;

        static const mo::TypeInfo _void_type_info;
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

        template <class T>
        inline void apply(bool* modified) const;
        template <class T>
        inline void apply(const bool modified);

      private:
        std::tuple<TInputParamPtr<Types>...> m_inputs;
        mo::IParam* m_current_input = nullptr;
    };
}

#define MULTI_INPUT(name, ...)                                                                                         \
    mo::TMultiInput<__VA_ARGS__> name##_param;                                                                         \
    typename mo::TMultiInput<__VA_ARGS__>::InputTypeTuple name;                                                        \
    VISIT(name, mo::INPUT, mo::TMultiInput<__VA_ARGS__>::initNullptr())
