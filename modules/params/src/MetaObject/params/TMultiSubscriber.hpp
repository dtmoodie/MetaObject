#ifndef MO_PARAMS_TMULTISUBSCRIBER_HPP
#define MO_PARAMS_TMULTISUBSCRIBER_HPP

#include "TParam.hpp"
#include "TSubscriberPtr.hpp"
#include "TypeSelector.hpp"

#include <tuple>
namespace mo
{
    class MO_EXPORTS IMultiSubscriber : public TParam<ISubscriber>
    {
      public:
        IMultiSubscriber();

        bool setInput(std::shared_ptr<IPublisher> input) override;
        bool setInput(IPublisher* input) override;

        // IDataContainerConstPtr_t getInputData(IAsyncStream* = nullptr) const override;

        IDataContainerConstPtr_t getCurrentData(IAsyncStream* = nullptr) const override;
        IDataContainerConstPtr_t getData(const Header* desired = nullptr, IAsyncStream* stream = nullptr) override;

        void setMtx(Mutex_t& mtx) override;

        mo::IPublisher* getPublisher() const override;

        bool isInputSet() const override;

        bool acceptsPublisher(const IPublisher& input) const override;

        bool acceptsType(const TypeInfo& type) const override;

        // Virtual to allow typed overload for interface slot input
        ConnectionPtr_t registerUpdateNotifier(ISlot& f) override;
        ConnectionPtr_t registerUpdateNotifier(const ISignalRelay::Ptr_t& relay) override;

        std::ostream& print(std::ostream&) const override;

        std::vector<Header> getAvailableHeaders() const override;
        boost::optional<Header> getNewestHeader() const override;

        bool hasNewData() const override;

      protected:
        void setInputs(const std::vector<ISubscriber*>& inputs);

      private:
        std::vector<ISubscriber*> m_inputs;
        ISubscriber* m_current_input = nullptr;
    };

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
} // namespace mo

#define MULTI_INPUT(name, ...)                                                                                         \
    mo::TMultiSubscriber<__VA_ARGS__> name##_param;                                                                    \
    typename mo::TMultiSubscriber<__VA_ARGS__>::InputTypeTuple name;                                                   \
    VISIT(name, mo::INPUT, mo::TMultiSubscriber<__VA_ARGS__>::initNullptr())

#endif // MO_PARAMS_TMULTISUBSCRIBER_HPP