#pragma once
#include <MetaObject/core/Demangle.hpp>
#include <MetaObject/params/ITAccessibleParam.hpp>
#include <MetaObject/params/MetaParam.hpp>
#include <MetaObject/params/ui/WT.hpp>
#include <MetaObject/params/ui/WidgetFactory.hpp>
#include <MetaObject/signals/TSlot.hpp>

#include <Wt/WContainerWidget>
#include <Wt/WText>

#include <boost/thread/recursive_mutex.hpp>
namespace Wt
{
    namespace Chart
    {
        class WAbstractChart;
    }
}
namespace mo
{
    class IParam;
    namespace UI
    {
        namespace wt
        {
            class MainApplication;
            class MO_EXPORTS IParamProxy : public Wt::WContainerWidget
            {
              public:
                IParamProxy(IParam* param_, MainApplication* app_, WContainerWidget* parent_ = 0);
                virtual ~IParamProxy();
                virtual void SetTooltip(const std::string& tip) = 0;

              protected:
                template <class T, class E>
                friend class TDataProxy;
                virtual void onParamUpdate(mo::Context* ctx, mo::IParam* param) = 0;
                virtual void onUiUpdate() = 0;
                mo::TSlot<void(mo::Context*, mo::IParam*)> _onUpdateSlot;
                std::shared_ptr<mo::Connection> _onUpdateConnection;
                MainApplication* _app;
            };

            class MO_EXPORTS IPlotProxy : public Wt::WContainerWidget
            {
              public:
                IPlotProxy(IParam* param_, MainApplication* app_, WContainerWidget* parent_ = 0);

                virtual ~IPlotProxy();
                virtual Wt::Chart::WAbstractChart* GetPlot() { return nullptr; }
              protected:
                template <class T, class E>
                friend class TDataProxy;
                virtual void onParamUpdate(mo::Context* ctx, mo::IParam* param) = 0;
                virtual void onUiUpdate() = 0;
                mo::TSlot<void(mo::Context*, mo::IParam*)> _onUpdateSlot;
                std::shared_ptr<mo::Connection> _onUpdateConnection;
                MainApplication* _app;
            };

            template <class T, class Enable = void>
            class TDataProxy
            {
              public:
                static const bool IS_DEFAULT = true;
                TDataProxy() {}
                void CreateUi(IParamProxy* proxy, T& data, bool read_only) {}
                void UpdateUi(const T& data) {}
                void onUiUpdate(T& data) {}
                void SetTooltip(const std::string& tooltip) {}
            };

            template <class T, typename Enable = void>
            class TParamProxy : public IParamProxy
            {
              public:
                static const bool IS_DEFAULT = TDataProxy<T, void>::IS_DEFAULT;

                TParamProxy(ITAccessibleParam<T>* param_, MainApplication* app_, WContainerWidget* parent_ = 0)
                    : IParamProxy(param_, app_, parent_), _param(param_), _data_proxy()
                {
                    auto token = param_->access();
                    _data_proxy.CreateUi(this, token(), param_->checkFlags(ParamFlags::State_e));
                }
                void SetTooltip(const std::string& tip) { _data_proxy.SetTooltip(tip); }
              protected:
                void onParamUpdate(mo::Context* ctx, mo::IParam* param)
                {
                    auto token = _param->access();
                    _app->getUpdateLock();
                    _data_proxy.UpdateUi(token());
                    _app->requestUpdate();
                }
                void onUiUpdate()
                {
                    auto token = _param->access();
                    _data_proxy.onUiUpdate(token());
                }
                mo::ITAccessibleParam<T>* _param;
                TDataProxy<T, void> _data_proxy;
            };

            template <class T, typename Enable = void>
            class TPlotDataProxy
            {
              public:
                static const bool IS_DEFAULT = true;
                TPlotDataProxy() {}
                void CreateUi(Wt::WContainerWidget* container, T& data, bool read_only, const std::string& name = "") {}
                void UpdateUi(const T& data, mo::Time_t ts) {}
                void onUiUpdate(T& data) {}
            };

            template <class T, typename Enable = void>
            class TPlotProxy : public IPlotProxy
            {
              public:
                static const bool IS_DEFAULT = TPlotDataProxy<T, void>::IS_DEFAULT;

                TPlotProxy(ITAccessibleParam<T>* param_, MainApplication* app_, WContainerWidget* parent_ = 0)
                    : IPlotProxy(param_, app_, parent_), _param(param_)
                {
                    auto token = param_->access();
                    if (IPlotProxy* parent = dynamic_cast<IPlotProxy*>(parent_))
                    {
                        _data_proxy.CreateUi(
                            parent, token(), param_->checkFlags(ParamFlags::State_e), param_->getTreeName());
                    }
                    else
                    {
                        _data_proxy.CreateUi(
                            this, token(), param_->checkFlags(ParamFlags::State_e), param_->getTreeName());
                    }
                }

              protected:
                void onParamUpdate(mo::Context* ctx, mo::IParam* param)
                {
                    auto token = _param->access();
                    _app->getUpdateLock();
                    // TODO FIX ME
                    _data_proxy.UpdateUi(token(), *_param->getTimestamp());
                    _app->requestUpdate();
                }
                void onUiUpdate()
                {
                    auto token = _param->access();
                    _data_proxy.onUiUpdate(token());
                }
                mo::ITAccessibleParam<T>* _param;
                TPlotDataProxy<T, void> _data_proxy;
            };

            template <class T>
            struct WidgetConstructor
            {

                WidgetConstructor()
                {
                    if (!TParamProxy<T, void>::IS_DEFAULT)
                        WidgetFactory::Instance()->RegisterConstructor(TypeInfo(typeid(T)),
                                                                       std::bind(WidgetConstructor<T>::CreateWidget,
                                                                                 std::placeholders::_1,
                                                                                 std::placeholders::_2,
                                                                                 std::placeholders::_3));
                }
                static IParamProxy* CreateWidget(IParam* param, MainApplication* app, Wt::WContainerWidget* container)
                {
                    if (param->getTypeInfo() == TypeInfo(typeid(T)))
                    {
                        auto typed = dynamic_cast<ITAccessibleParam<T>*>(param);
                        if (typed)
                        {
                            return new TParamProxy<T, void>(typed, app, container);
                        }
                    }
                    return nullptr;
                }
            };
            template <class T>
            struct PlotConstructor
            {
                PlotConstructor()
                {
                    if (!TParamProxy<T, void>::IS_DEFAULT)
                        WidgetFactory::Instance()->RegisterConstructor(TypeInfo(typeid(T)),
                                                                       std::bind(&PlotConstructor<T>::CreatePlot,
                                                                                 std::placeholders::_1,
                                                                                 std::placeholders::_2,
                                                                                 std::placeholders::_3));
                }
                static IPlotProxy* CreatePlot(IParam* param, MainApplication* app, Wt::WContainerWidget* container)
                {
                    if (param->getTypeInfo() == TypeInfo(typeid(T)))
                    {
                        auto typed = dynamic_cast<ITAccessibleParam<T>*>(param);
                        if (typed)
                        {
                            return new TPlotProxy<T, void>(typed, app, container);
                        }
                    }
                    return nullptr;
                }
            };
        }
    }
#define MO_UI_WT_PARAMTERPROXY_METAParam(N)                                                                            \
    template <class T>                                                                                                 \
    struct MetaParam<T, N, typename std::enable_if<!mo::UI::wt::TParamProxy<T>::IS_DEFAULT>::type>                     \
        : public MetaParam<T, N - 1, void>                                                                             \
    {                                                                                                                  \
        static UI::wt::WidgetConstructor<T> _Param_proxy_constructor;                                                  \
        MetaParam(const char* name) : MetaParam<T, N - 1, void>(name) { (void)&_Param_proxy_constructor; }             \
    };                                                                                                                 \
    template <class T>                                                                                                 \
    UI::wt::WidgetConstructor<T>                                                                                       \
        MetaParam<T,                                                                                                   \
                  N,                                                                                                   \
                  typename std::enable_if<!mo::UI::wt::TParamProxy<T>::IS_DEFAULT>::type>::_Param_proxy_constructor;

    MO_UI_WT_PARAMTERPROXY_METAParam(__COUNTER__)

#define MO_UI_WT_PLOTPROXY_METAParam(N)                                                                                \
    template <class T>                                                                                                 \
    struct MetaParam<T, N, typename std::enable_if<!mo::UI::wt::TPlotProxy<T>::IS_DEFAULT>::type>                      \
        : public MetaParam<T, N - 1, void>                                                                             \
    {                                                                                                                  \
        static UI::wt::PlotConstructor<T> _Param_plot_constructor;                                                     \
        MetaParam(const char* name) : MetaParam<T, N - 1, void>(name) { (void)&_Param_plot_constructor; }              \
    };                                                                                                                 \
    template <class T>                                                                                                 \
    UI::wt::PlotConstructor<T>                                                                                         \
        MetaParam<T,                                                                                                   \
                  N,                                                                                                   \
                  typename std::enable_if<!mo::UI::wt::TPlotProxy<T>::IS_DEFAULT>::type>::_Param_plot_constructor;

        MO_UI_WT_PLOTPROXY_METAParam(__COUNTER__)
}
