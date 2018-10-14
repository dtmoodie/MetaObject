#ifndef TMULTIOUTPUT_HPP
#define TMULTIOUTPUT_HPP
#include "MetaObject/params/IParam.hpp"
#include "OutputParam.hpp"
#include "TMultiInput-inl.hpp"
#include "TMultiInput.hpp"
#include "TParam.hpp"

namespace mo
{
    struct IMultiOutput : public OutputParam
    {
        IMultiOutput(const std::vector<IParam*>& outputs);
        virtual std::vector<TypeInfo> listOutputTypes() const override;

        ParamBase* getOutputParam(const TypeInfo type) override;
        const ParamBase* getOutputParam(const TypeInfo type) const override;

        ParamBase* getOutputParam() override;
        const ParamBase* getOutputParam() const override;

        void setName(const std::string& name);

        bool providesOutput(const TypeInfo type) const override;

        TypeInfo getTypeInfo() const override;

        std::ostream& print(std::ostream& os) const override;

        virtual void visit(IReadVisitor*) override;
        virtual void visit(IWriteVisitor*) const override;

        IContainerPtr_t getData(const Header& header);
        IContainerConstPtr_t getData(const Header& header) const;

      private:
        std::vector<IParam*> m_outputs;
        TypeInfo m_current_type = TypeInfo::Void();
    };

    template <class... Types>
    struct TMultiOutput : public IMultiOutput
    {
      public:
        TMultiOutput()
            : IMultiOutput(globParamPtrs<IParam>(m_params))
        {
        }
        template <class T, class... Args>
        void updateData(T&& data, Args&&... args)
        {
            get<TParam<typename std::decay<T>::type>>(m_params).updateData(std::move(data),
                                                                           std::forward<Args>(args)...);
            m_current_type = TypeInfo(typeid(T));
        }

      private:
        std::tuple<Types...> m_data;
        std::tuple<TParam<Types>...> m_params;
    };
}
#endif // TMULTIOUTPUT_HPP
