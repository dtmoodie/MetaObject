#ifndef TMULTIOUTPUT_HPP
#define TMULTIOUTPUT_HPP
#include "MetaObject/params/IParam.hpp"
#include "OutputParam.hpp"
#include "TParamPtr.hpp"
#include "TMultiInput.hpp"

namespace mo
{
    template<class ... Types>
    class TMultiOutput: virtual public OutputParam
    {
    public:


        std::vector<TypeInfo> listOutputTypes() const override{
            return{TypeInfo(typeid(Types))...};
        }

        template<class T>
        void apply(IParam** param){
            *param = &get<TParamPtr<T>>(m_params);
        }

        IParam* getOutputParam(const TypeInfo type) override{
            IParam* out=nullptr;
            selectType(*this, &out);
            return out;
        }

        template<class T, class...Args>
        void updateData(T&& data, Args&&... args){
            get<TParamPtr<typename std::decay<T>::type>>(m_params)
                .updateData(std::move(data), std::move<Args>(args)...);
        }
    private:
        std::tuple<Types...> m_data;
        std::tuple<TParamPtr<Types>...> m_params;
    };
}
#endif // TMULTIOUTPUT_HPP
