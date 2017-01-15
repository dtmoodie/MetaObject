#include <MetaObject/MetaObject.hpp>
#include <MetaObject/Parameters/Demangle.hpp>
#include <MetaObject/Parameters/Types.hpp>

#include <Wt/WApplication>
#include <Wt/WBreak>
#include <Wt/WContainerWidget>
#include <Wt/WLineEdit>
#include <Wt/WPushButton>
#include <Wt/WText>
#include <Wt/WSlider>
#include <Wt/WSpinBox>
#include <Wt/WComboBox>

#include <boost/thread.hpp>

#include <functional>

using namespace Wt;




namespace mo
{
    namespace UI
    {
        namespace wt
        {
            class IParameterProxy;
            class IParameterInputProxy;
            class IParameterOutputProxy;
            class MainApplication : public WApplication
            {
            public:
                MainApplication(const Wt::WEnvironment& env, ITypedParameter<std::string>* param);
                void requestUpdate()
                {
                    _dirty = true;
                    auto current_time = boost::posix_time::microsec_clock::universal_time();
                    if((current_time - _last_update_time).total_milliseconds() > 15)
                    {
                        this->triggerUpdate();
                    }
                }
            private:
                Wt::WLineEdit *nameEdit_;
                Wt::WText *greeting_;

                void greet();
                bool _dirty;
                boost::posix_time::ptime _last_update_time;
            };
            class WidgetFactory
            {
            public:
                typedef std::function<IParameterProxy*(mo::IParameter*)> WidgetConstructor_f;

                static WidgetFactory* Instance();
                IParameterProxy* CreateWidget(mo::IParameter* param);
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
            IParameterProxy* WidgetFactory::CreateWidget(mo::IParameter* param)
            {
                if(param->CheckFlags(mo::Input_e))
                    return nullptr;
                if(param->CheckFlags(mo::Output_e))
                    return nullptr;
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
        



            



            

        }
    }

}




void mo::UI::wt::MainApplication::greet()
{
    greeting_->setText("Hello there, " + nameEdit_->text());
}

WApplication *createApplication(const WEnvironment& env, mo::ITypedParameter<std::string>* param)
{
    return new mo::UI::wt::MainApplication(env, param);
}

int main(int argc, char** argv)
{
    std::string str;
    mo::TypedParameterPtr<std::string> param;
    param.UpdatePtr(&str);
    param.SetName("alskdfjlakjsdflkasjdfklj");
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
