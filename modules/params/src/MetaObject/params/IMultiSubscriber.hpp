#ifndef MO_PARAMS_IMULTISUBSCRIBER_HPP
#define MO_PARAMS_IMULTISUBSCRIBER_HPP

#include "TParam.hpp"

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

        TypeInfo getCurrentInputType() const;

      protected:
        void setInputs(const std::vector<ISubscriber*>& inputs);

      private:
        std::vector<ISubscriber*> m_inputs;
        ISubscriber* m_current_input = nullptr;
    };
} // namespace mo
#endif // MO_PARAMS_IMULTISUBSCRIBER_HPP