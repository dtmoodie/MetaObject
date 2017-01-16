#pragma once
#include <MetaObject/Parameters/MetaParameter.hpp>
#include <MetaObject/Parameters/UI/WidgetFactory.hpp>
#include <MetaObject/Parameters/UI/WT.hpp>
#include <MetaObject/Signals/TypedSlot.hpp>
#include <MetaObject/Parameters/Demangle.hpp>
#include <MetaObject/Parameters/IParameter.hpp>

#include <Wt/WContainerWidget>
#include <Wt/WText>

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
                virtual void onUpdate(mo::Context* ctx, mo::IParameter* param) = 0;
                mo::TypedSlot<void(mo::Context*, mo::IParameter*)> _onUpdateSlot;
                std::shared_ptr<mo::Connection>  _onUpdateConnection;
                MainApplication* _app;
            };

            template<class T, class enable = void>
            class TParameterProxy : public IParameterProxy
            {
            public:
                static const bool IS_DEFAULT = true;
                TParameterProxy(ITypedParameter<T>* null)
                {
                    (void)null;
                }
                void SetTooltip(const std::string& tip){ (void)tip; }
            protected:
                void onUpdate(mo::Context* ctx, mo::IParameter* param){ (void) param; }
            };

            template<class T>
            struct Void {
                typedef void type;
            };

            template<class T, class U = void>
            struct is_default {
                enum { value = 0 };
            };

            template<class T>
            struct is_default<T, typename Void<typename T::IsDefault>::type > {
                enum { value = 1 };
            };
            template<class T> struct Constructor
            {
            public:
                Constructor()
                {
                    WidgetFactory::Instance()->RegisterConstructor(TypeInfo(typeid(T)),
                                std::bind(Constructor<T>::Create, std::placeholders::_1, std::placeholders::_2));
                }
                static IParameterProxy* Create(IParameter* param, MainApplication* app)
                {
                    if (param->GetTypeInfo() == TypeInfo(typeid(T)))
                    {
                        auto typed = dynamic_cast<ITypedParameter<T>*>(param);
                        if (typed)
                        {
                             return new TParameterProxy<T, void>(typed, app);
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
