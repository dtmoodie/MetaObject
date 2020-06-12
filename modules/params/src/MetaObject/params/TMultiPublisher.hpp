#ifndef TMULTIOUTPUT_HPP
#define TMULTIOUTPUT_HPP
#include "IPublisher.hpp"
#include "TMultiSubscriber.hpp"

#include <MetaObject/params/IParam.hpp>
#include <MetaObject/params/TPublisher.hpp>
namespace mo
{
    struct MO_EXPORTS IMultiPublisher : public TParam<IPublisher>
    {
        IMultiPublisher();
        virtual std::vector<TypeInfo> getOutputTypes() const override;

        void setName(std::string name) override;

        bool providesOutput(const TypeInfo type) const override;

        std::vector<Header> getAvailableHeaders() const override;
        boost::optional<Header> getNewestHeader() const override;

        std::ostream& print(std::ostream& os) const override;

        void load(ILoadVisitor&) override;
        void save(ISaveVisitor&) const override;
        void visit(StaticVisitor&) const override;

        IDataContainerConstPtr_t getData(const Header* desired = nullptr, IAsyncStream* = nullptr) override;

        uint32_t getNumSubscribers() const override;
        void setAllocator(Allocator::Ptr_t) override;

      protected:
        void setOutputs(std::vector<IPublisher*>&& outputs);
        IPublisher* getPublisher(const TypeInfo type);
        const IPublisher* getPublisher(const TypeInfo type) const;
        IPublisher* getPublisher();
        const IPublisher* getPublisher() const;

      private:
        std::vector<IPublisher*> m_outputs;
        TypeInfo m_current_type = TypeInfo::Void();
    };

    template <class... Types>
    struct TMultiOutput : public IMultiPublisher
    {
      public:
        TMultiOutput()
        {
            IMultiPublisher::setOutputs(globParamPtrs<IPublisher>(m_params));
        }

        template <class T, class... Args>
        void publish(T&& data, Args&&... args)
        {
            using type = typename std::decay<T>::type;
            get<TPublisher<type>>(m_params).publish(std::move(data), std::forward<Args>(args)...);
            m_current_type = TypeInfo(typeid(T));
        }

      private:
        std::tuple<TPublisher<Types>...> m_params;
    };
} // namespace mo
#endif // TMULTIOUTPUT_HPP
