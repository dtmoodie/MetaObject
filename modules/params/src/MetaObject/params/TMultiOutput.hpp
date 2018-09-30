#ifndef TMULTIOUTPUT_HPP
#define TMULTIOUTPUT_HPP
#include "MetaObject/params/IParam.hpp"
#include "OutputParam.hpp"
#include "TMultiInput.hpp"
#include "TParam.hpp"

namespace mo
{
    template <class... Types>
    class TMultiOutput : public OutputParam
    {
      public:
        TMultiOutput() { this->setFlags(mo::ParamFlags::Output_e); }
        std::vector<TypeInfo> listOutputTypes() const override { return {TypeInfo(typeid(Types))...}; }

        template <class T>
        void apply(ParamBase** param)
        {
            *param = &mo::get<TParam<T>>(m_params);
        }

        template <class T>
        void apply(const ParamBase** param) const
        {
            *param = &mo::get<TParam<T>>(m_params);
        }

        template <class T>
        void apply(const std::string& name)
        {
            mo::get<TParam<T>>(m_params).setName(name);
        }

        template <class T>
        void apply(std::ostream& os) const
        {
            mo::get<TParam<T>>(m_params).print(os);
        }

        ParamBase* getOutputParam(const TypeInfo type) override
        {
            ParamBase* out = nullptr;
            selectType<Types...>(*this, type, &out);
            return out;
        }

        const ParamBase* getOutputParam(const TypeInfo type) const override
        {
            const ParamBase* out = nullptr;
            selectType<Types...>(*this, type, &out);
            return out;
        }

        ParamBase* getOutputParam() override { return getOutputParam(current_type); }

        const ParamBase* getOutputParam() const override { return getOutputParam(current_type); }

        template<class T>
        void apply(const TypeInfo type, bool* success) const
        {
            if(type == TypeInfo(typeid(T)))
            {
                *success = true;
            }
        }

        bool providesOutput(const TypeInfo type) const
        {
            bool success = false;
            typeLoop<Types...>(*this, type, &success);
            return success;
        }

        void setName(const std::string& name)
        {
            OutputParam::setName(name);
            typeLoop<Types...>(*this, name);
        }

        template <class T, class... Args>
        void updateData(T&& data, Args&&... args)
        {
            get<TParam<typename std::decay<T>::type>>(m_params).updateData(std::move(data),
                                                                           std::forward<Args>(args)...);
            current_type = TypeInfo(typeid(T));
        }

        TypeInfo getTypeInfo() const override { return current_type; }

        std::ostream& print(std::ostream& os) const override
        {
            OutputParam::print(os);
            selectType<Types...>(*this, current_type, os);
            return os;
        }

      private:
        std::tuple<Types...> m_data;
        std::tuple<TParam<Types>...> m_params;
        TypeInfo current_type = TypeInfo(typeid(void));
    };
}
#endif // TMULTIOUTPUT_HPP
