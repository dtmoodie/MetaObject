#pragma once

#include "IParameterProxy.hpp"
#include <MetaObject/Parameters/ITypedParameter.hpp>
#include <Wt/WSpinBox>
namespace mo
{
namespace UI
{
namespace wt
{
    template<class T>
    class TDataProxy<T, typename std::enable_if<std::is_integral<T>::value && !std::is_same<T, bool>::value>::type>
    {
    public:
        static const bool IS_DEFAULT = false;
        TDataProxy(IParameterProxy& proxy):
            _spin_box(nullptr),
            _proxy(proxy)
        {

        }
        void CreateUi(IParameterProxy* proxy, T* data)
        {
            _spin_box = new Wt::WSpinBox(proxy);
            if(data)
            {
                _spin_box->setValue(*data);
                _spin_box->changed().connect(proxy, &IParameterProxy::onUiUpdate);
            }

        }
        void UpdateUi(const T& data)
        {
            _spin_box->setValue(data);
        }
        void onUiUpdate(T& data)
        {
            data = _spin_box->value();
        }
        void SetTooltip(const std::string& tooltip){}
    protected:
        IParameterProxy& _proxy;
        Wt::WSpinBox* _spin_box;
    };

    template<>
    class MO_EXPORTS TDataProxy<bool, void>
    {
    public:
        static const bool IS_DEFAULT = false;
        TDataProxy(IParameterProxy &proxy);
        void CreateUi(IParameterProxy* proxy, bool* data);
        void UpdateUi(const bool& data);
        void onUiUpdate(bool& data);
        void SetTooltip(const std::string& tooltip);
    protected:
        Wt::WCheckBox* _check_box;
        IParameterProxy& _proxy;
    };
} // namespace wt
} // namespace UI
} // namespace mo
