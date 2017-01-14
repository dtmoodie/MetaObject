#include <MetaObject/MetaObject.hpp>
#include <MetaObject/Parameters/Demangle.hpp>

#include <Wt/WApplication>
#include <Wt/WBreak>
#include <Wt/WContainerWidget>
#include <Wt/WLineEdit>
#include <Wt/WPushButton>
#include <Wt/WText>

#include <boost/thread.hpp>

#include <functional>

using namespace Wt;

namespace mo
{
    namespace UI
    {
        namespace wt
        {
            class WidgetFactory
            {
            public:
                typedef std::function<Wt::WWidget*(mo::IParameter*)> WidgetConstructor_f;

                static WidgetFactory* Instance();
                Wt::WWidget* CreateWidget(mo::IParameter* param);
                void RegisterConstructor(const mo::TypeInfo& type, const WidgetConstructor_f& constructor);
            private:
                std::map<mo::TypeInfo, WidgetConstructor_f> _constructors;
            };

            WidgetFactory* WidgetFactory::Instance()
            {
                static WidgetFactory* g_inst = nullptr;
                if(g_inst == nullptr)
                    g_inst = new WidgetFactory();
                return g_inst;
            }
            Wt::WWidget* WidgetFactory::CreateWidget(mo::IParameter* param)
            {
                auto itr = _constructors.find(param->GetTypeInfo());
                if(itr != _constructors.end())
                {
                    return itr->second(param);
                }
                return nullptr;
            }
            void WidgetFactory::RegisterConstructor(const mo::TypeInfo& type, const WidgetConstructor_f& constructor)
            {
                if(_constructors.find(type) == _constructors.end())
                {
                    _constructors[type] = constructor;
                }
            }
        
            template<class T, class enable = void>
            class TParameterProxy: public Wt::WContainerWidget
            {
                typedef void IsDefault;
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

            template<class T>
            class TParameterProxy<T, typename std::enable_if<std::is_pod<T>::value>::type>
            {
            public:


            private:
                ITypedParameter<T>* param;
            };

            template<>
            class TParameterProxy<std::string, void>: public Wt::WContainerWidget
            {
            public:
                TParameterProxy(ITypedParameter<std::string>* param_, Wt::WApplication* app,
                                WContainerWidget *parent = 0):
                    Wt::WContainerWidget(parent),
                    _param(param_),
                    _app(app),
                    _onUpdateSlot(std::bind(&TParameterProxy<std::string, void>::onUpdate, this, std::placeholders::_1, std::placeholders::_2))
                {
                    std::stringstream ss;
                    ss << param_->GetTreeName() << "[" << mo::Demangle::TypeToName(param_->GetTypeInfo())
                       << "]";
                    this->addWidget(new Wt::WText(ss.str(), this));
                    _line_edit = new Wt::WLineEdit(this);
                    _line_edit->setText(param_->GetData());
                    _onUpdateConnection = param_->RegisterUpdateNotifier(&_onUpdateSlot);
                }

            private:
                ITypedParameter<std::string>* _param;
                Wt::WLineEdit* _line_edit;
                void onUpdate( mo::Context* ctx, mo::IParameter* param)
                {
                    auto lock = _app->getUpdateLock();
                    _line_edit->setText(_param->GetData());
                    _app->triggerUpdate();
                }

                mo::TypedSlot<void(mo::Context*, mo::IParameter*)> _onUpdateSlot;
                std::shared_ptr<mo::Connection>  _onUpdateConnection;
                Wt::WApplication* _app;
            };

            
            template<class T> struct Constructor
            {
            public:
                Constructor()
                {
                    WidgetFactory::Instance()->RegisterConstructor(TypeInfo(typeid(T)), std::bind(Constructor<T>::Create, std::placeholders::_1));
                }
                static Wt::WWidget* Create(IParameter* param)
                {
                    if (param->GetTypeInfo() == TypeInfo(typeid(T)))
                    {
                        auto typed = static_cast<ITypedParameter<T>*>(param);
                        if (typed)
                        {
                            return TParameterProxy<T>(typed);
                        }
                    }
                    return nullptr;
                }
            };
        }
    }
#define MO_UI_WT_PARAMTERPROXY_METAPARAMETER(N) \
    template<class T> struct MetaParameter<T, N, std::enable_if<UI::wt::is_default<mo::UI::wt::TParameterProxy<T>>::value> : public MetaParameter<T, N - 1, void> \
    { \
        static UI::wt::Constructor<T> _parameter_proxy_constructor; \
        MetaParameter(const char* name): \
            MetaParameter<T, N-1, void>(name) \
        { \
            (void)&_parameter_proxy_constructor; \
        } \
    }; \
    template<class T> UI::wt::Constructor<T> MetaParameter<T,N, void>::_parameter_proxy_constructor;
}





class MainApplication: public WApplication
{
public:
    MainApplication(const Wt::WEnvironment& env, mo::ITypedParameter<std::string>* param);

private:
    Wt::WLineEdit *nameEdit_;
    Wt::WText *greeting_;
    
    void greet();
};

MainApplication::MainApplication(const WEnvironment& env, mo::ITypedParameter<std::string>* param)
    : WApplication(env)
{
    setTitle("Hello world");                               // application title
    enableUpdates();

    root()->addWidget(new WText("Your name, please ? "));  // show some text
    nameEdit_ = new WLineEdit(root());                     // allow text input
    nameEdit_->setFocus();                                 // give focus

    //auto proxy = mo::UI::wt::WidgetFactory::Instance()->CreateWidget(param);
    auto proxy = new mo::UI::wt::TParameterProxy<std::string>(param, this);
    root()->addWidget(proxy);

    WPushButton *button
        = new WPushButton("Greet me.", root());              // create a button
    button->setMargin(5, Left);                            // add 5 pixels margin

    root()->addWidget(new WBreak());                       // insert a line break
    
    greeting_ = new WText(root());                         
    button->clicked().connect(this, &MainApplication::greet);
    nameEdit_->enterPressed().connect(
                boost::bind(&MainApplication::greet, this));
}

void MainApplication::greet()
{
    greeting_->setText("Hello there, " + nameEdit_->text());
}

WApplication *createApplication(const WEnvironment& env, mo::ITypedParameter<std::string>* param)
{
    return new MainApplication(env, param);
}
int main(int argc, char** argv)
{
    std::string str;
    mo::TypedParameterPtr<std::string> param;
    param.UpdatePtr(&str);
    boost::thread processingthread(std::bind(
    [&str, &param]()
    {
        int tick = 0;
        while(!boost::this_thread::interruption_requested())
        {
            boost::this_thread::sleep_for(boost::chrono::milliseconds(100));
            param.UpdateData(boost::lexical_cast<std::string>(tick));
            ++tick;
        }
    }));
    return WRun(argc, argv, std::bind(&createApplication, std::placeholders::_1, &param));
}
