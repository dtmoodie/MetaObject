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

        bool acceptsPublisher(const IPublisher& param) const override;
        bool acceptsType(const mo::TypeInfo& type) const override;

      protected:
    };
} // namespace mo

#endif // MO_PARAMS_SUBSCRIBERANY_HPP