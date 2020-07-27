#ifndef MO_PARAMS_SUBSCRIBERANY_HPP
#define MO_PARAMS_SUBSCRIBERANY_HPP
#include "ISubscriber.hpp"
#include "TParam.hpp"
namespace mo
{
    class MO_EXPORTS SubscriberAny : public TParam<ISubscriber>
    {
      public:
        SubscriberAny(const std::string& name = "");

        bool setInput(std::shared_ptr<IPublisher> param) override;
        bool setInput(IPublisher* param = nullptr) override;

        bool acceptsPublisher(const IPublisher& param) const override;
        bool acceptsType(const mo::TypeInfo& type) const override;

        std::vector<TypeInfo> getInputTypes() const override;
        bool isInputSet() const override;

        IDataContainerConstPtr_t getCurrentData(IAsyncStream* = nullptr) const override;

        bool hasNewData() const override;
        boost::optional<Header> getNewestHeader() const override;

        IDataContainerConstPtr_t getData(const Header* desired = nullptr, IAsyncStream* stream = nullptr) override;

        std::vector<Header> getAvailableHeaders() const override;

        std::ostream& print(std::ostream&) const override;
        IPublisher* getPublisher() const override;

      protected:
        IPublisher* m_publisher = nullptr;
        std::shared_ptr<IPublisher> m_shared_publisher;
    };
} // namespace mo

#endif // MO_PARAMS_SUBSCRIBERANY_HPP