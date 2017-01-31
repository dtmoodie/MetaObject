#pragma once
#include <MetaObject/Parameters/MetaParameter.hpp>
#include <MetaObject/Parameters/UI/WidgetFactory.hpp>
#include <MetaObject/Parameters/UI/WT.hpp>
#include <MetaObject/Signals/TypedSlot.hpp>
#include <MetaObject/Parameters/Demangle.hpp>
#include <MetaObject/Parameters/IParameter.hpp>

#include <Wt/WContainerWidget>
#include <Wt/WText>

#include <boost/thread/recursive_mutex.hpp>

namespace mo
{
class IParameter;
namespace UI
{
namespace wt
{
    class MainApplication;
    class MO_EXPORTS IParameterProxy : public Wt::WContainerWidget
    {
    public:
        IParameterProxy(IParameter* param_, MainApplication* app_,
            WContainerWidget *parent_ = 0);
        virtual ~IParameterProxy();
        virtual void SetTooltip(const std::string& tip) = 0;
    protected:
        template<class T, class E> friend class TDataProxy;
        virtual void onParameterUpdate(mo::Context* ctx, mo::IParameter* param) = 0;
        virtual void onUiUpdate() = 0;
        mo::TypedSlot<void(mo::Context*, mo::IParameter*)> _onUpdateSlot;
        std::shared_ptr<mo::Connection>  _onUpdateConnection;
        MainApplication* _app;
    };

    template<class T, class Enable = void>
    class TDataProxy
    {
    public:
        static const bool IS_DEFAULT = true;
        TDataProxy(IParameterProxy& proxy){}
        void CreateUi(IParameterProxy* proxy, T* data){}
        void UpdateUi(const T& data){}
        void onUiUpdate(T& data){}
        void SetTooltip(const std::string& tooltip){}
    protected:
        IParameterProxy& _proxy;
    };

    template<class T, class enable = void>
    class TParameterProxy : public IParameterProxy
    {
    public:
        static const bool IS_DEFAULT = TDataProxy<T,void>::IS_DEFAULT;
        TParameterProxy(ITypedParameter<T>* param_, MainApplication* app_,
                        WContainerWidget *parent_ = 0):
            IParameterProxy(param_, app_, parent_),
            _param(param_),
            _data_proxy(*this)
        {
            boost::recursive_mutex::scoped_lock param_lock(_param->mtx());
            T* ptr = param_->GetDataPtr();
            if(ptr)
            {
                _data_proxy.CreateUi(this, ptr);
            }
        }
        void SetTooltip(const std::string& tip)
        {
            _data_proxy.SetTooltip(tip);
        }
    protected:
        void onParameterUpdate(mo::Context* ctx, mo::IParameter* param)
        {
            boost::recursive_mutex::scoped_lock param_lock(_param->mtx());
            T* ptr = _param->GetDataPtr();
            if(ptr)
            {
                _app->getUpdateLock();
                _data_proxy.UpdateUi(*ptr);
                _app->requestUpdate();
            }
        }
        void onUiUpdate()
        {
            boost::recursive_mutex::scoped_lock param_lock(_param->mtx());
            T* ptr = _param->GetDataPtr();
            if(ptr)
            {
                _data_proxy.onUiUpdate(*ptr);
                _param->Commit();
            }
        }
        mo::ITypedParameter<T>* _param;
        TDataProxy<T, void> _data_proxy;
    };

    template<class T> struct Constructor
    {
    public:
        Constructor()
        {
            WidgetFactory::Instance()->RegisterConstructor(TypeInfo(typeid(T)),
                        std::bind(Constructor<T>::Create, std::placeholders::_1,
                                  std::placeholders::_2, std::placeholders::_3));
        }
        static IParameterProxy* Create(IParameter* param, MainApplication* app, Wt::WContainerWidget* container)
        {
            if (param->GetTypeInfo() == TypeInfo(typeid(T)))
            {
                auto typed = dynamic_cast<ITypedParameter<T>*>(param);
                if (typed)
                {
                     return new TParameterProxy<T, void>(typed, app, container);
                }
            }
            return nullptr;
        }
    };
}
}
#define MO_UI_WT_PARAMTERPROXY_METAPARAMETER(N) \
template<class T> \
struct MetaParameter<T, N, typename std::enable_if<!mo::UI::wt::TParameterProxy<T>::IS_DEFAULT>::type> : public MetaParameter<T, N - 1, void> \
{ \
    static UI::wt::Constructor<T> _parameter_proxy_constructor; \
    MetaParameter(const char* name): \
        MetaParameter<T, N-1, void>(name) \
    { \
        (void)&_parameter_proxy_constructor; \
    } \
}; \
template<class T> UI::wt::Constructor<T> MetaParameter<T,N, typename std::enable_if<!mo::UI::wt::TParameterProxy<T>::IS_DEFAULT>::type>::_parameter_proxy_constructor;

MO_UI_WT_PARAMTERPROXY_METAPARAMETER(__COUNTER__)
}
