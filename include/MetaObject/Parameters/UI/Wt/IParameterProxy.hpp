#pragma once
#include <MetaObject/Parameters/MetaParameter.hpp>
#include <MetaObject/Parameters/UI/WidgetFactory.hpp>
#include <MetaObject/Parameters/UI/WT.hpp>
#include <MetaObject/Signals/TypedSlot.hpp>
#include <MetaObject/Parameters/Demangle.hpp>

#include <Wt/WContainerWidget>
#include <Wt/WText>

namespace mo
{
    IParameter;
    namespace UI
    {
        namespace wt
        {
            class MO_EXPORTS IParameterProxy : public Wt::WContainerWidget
            {
            public:
                static IParameterProxy* Create();
                IParameterProxy(IParameter* param_, MainApplication* app_,
                    WContainerWidget *parent_ = 0);
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
                typedef void IsDefault;
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
                    WidgetFactory::Instance()->RegisterConstructor(TypeInfo(typeid(T)), std::bind(Constructor<T>::Create, std::placeholders::_1));
                }
                static IParameterProxy* Create(IParameter* param)
                {
                    if (param->GetTypeInfo() == TypeInfo(typeid(T)))
                    {
                        auto typed = dynamic_cast<ITypedParameter<T>*>(param);
                        if (typed)
                        {
                             return new TParameterProxy<T, void>(typed);
                        }
                    }
                    return nullptr;
                }
            };
        }
    }
#define MO_UI_WT_PARAMTERPROXY_METAPARAMETER(N) \
    template<class T> \
    struct MetaParameter<T, N, typename std::enable_if<!UI::wt::is_default<mo::UI::wt::TParameterProxy<T>>::value>::type> : public MetaParameter<T, N - 1, void> \
    { \
        static UI::wt::Constructor<T> _parameter_proxy_constructor; \
        MetaParameter(const char* name): \
            MetaParameter<T, N-1, void>(name) \
        { \
            (void)&_parameter_proxy_constructor; \
        } \
    }; \
    template<class T> UI::wt::Constructor<T> MetaParameter<T,N, typename std::enable_if<!UI::wt::is_default<mo::UI::wt::TParameterProxy<T>>::value>::type>::_parameter_proxy_constructor;

    MO_UI_WT_PARAMTERPROXY_METAPARAMETER(__COUNTER__)
}