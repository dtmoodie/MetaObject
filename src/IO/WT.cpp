#include "MetaObject/IO/WT.hpp"

using namespace mo;
using namespace mo::IO;
using namespace mo::IO::web;
#ifdef HAVE_WT
#include <boost/thread.hpp>
#include <Wt/WApplication>
#include <Wt/WServer>
#include <Wt/WBreak>
#include <Wt/WContainerWidget>
#include <Wt/WLineEdit>
#include <Wt/WPushButton>
#include <Wt/WText>

class MetaObjectApplication: public Wt::WApplication
{
public:
    MetaObjectApplication(const Wt::WEnvironment& env):
        Wt::WApplication(env)
    {
    
    }
};

Wt::WApplication *createApplication(const Wt::WEnvironment& env)
{
    Wt::WApplication *app = new MetaObjectApplication(env);
    return app;
}


struct WebContext::impl
{
    Wt::WServer* server;
    void Run()
    {
        server = Wt::WServer::instance();
        server->start();
    }
    void Kill()
    {
        server->stop();
    }
};
#endif

WebContext* WebContext::Instance()
{
    static WebContext* g_ctx = nullptr;
    if(g_ctx == nullptr)
        g_ctx = new WebContext();
    return g_ctx;
}
WebContext::WebContext()
{
    _pimpl = new impl();
}
void WebContext::Start()
{
    _pimpl->Run();
}
void WebContext::Stop()
{
    _pimpl->Kill();
}