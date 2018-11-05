#pragma once
#include "TInputParam.hpp"
#include "TypeSelector.hpp"
#include <tuple>
namespace mo
{
    class IMultiInput : public InputParam
    {
      public:
        IMultiInput();

        virtual bool setInput(const std::shared_ptr<IParam>& input) override;
        virtual bool setInput(IParam* input) override;

        virtual IContainerPtr_t getData(const Header& desired = Header()) override;
        virtual IContainerConstPtr_t getData(const Header& desired = Header()) const override;

        virtual void setMtx(Mutex_t* mtx) override;

        mo::TypeInfo getTypeInfo() const override;
        mo::IParam* getInputParam() const override;

        OptionalTime getInputTimestamp() override;
        virtual uint64_t getInputFrameNumber() override;

        virtual OptionalTime getTimestamp() const override;
        virtual uint64_t getFrameNumber() const override;

        bool isInputSet() const override;

        bool acceptsInput(IParam* input) const override;

        bool acceptsType(const TypeInfo& type) const override;

        // Virtual to allow typed overload for interface slot input
        virtual ConnectionPtr_t registerUpdateNotifier(ISlot* f) override;
        virtual ConnectionPtr_t registerUpdateNotifier(const ISignalRelay::Ptr& relay) override;

        virtual bool modified() const override;
        virtual void modified(bool value) override;

      protected:
        void setInputs(const std::vector<InputParam*>& inputs);

      private:
        std::vector<InputParam*> m_inputs;
        InputParam* m_current_input = nullptr;

        static const mo::TypeInfo _void_type_info;
    };

    template <class... Types>
    class TMultiInput : virtual public IMultiInput
    {
      public:
        using InputTypeTuple = std::tuple<const Types*...>;
        using TypeTuple = std::tuple<Types...>;
        static InputTypeTuple initNullptr();

        TMultiInput();

        void setUserDataPtr(std::tuple<const Types*...>* user_var_);

        template <class T>
        inline void apply(std::tuple<const Types*...>* user_var_);

      private:
        void onInputUpdate(const IDataContainerPtr_t&, IParam*, UpdateFlags);
        std::tuple<TInputParamPtr<Types>...> m_inputs;
    };
}

#define MULTI_INPUT(name, ...)                                                                                         \
    mo::TMultiInput<__VA_ARGS__> name##_param;                                                                         \
    typename mo::TMultiInput<__VA_ARGS__>::InputTypeTuple name;                                                        \
    VISIT(name, mo::INPUT, mo::TMultiInput<__VA_ARGS__>::initNullptr())
