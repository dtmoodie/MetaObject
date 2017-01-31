#ifdef HAVE_WT
#include "MetaObject/Parameters/UI/WT.hpp"

using namespace mo;
using namespace mo::UI;
using namespace Wt;

#include <boost/thread.hpp>
#include <Wt/WApplication>
#include <Wt/WServer>
#include <Wt/WBreak>
#include <Wt/WContainerWidget>
#include <Wt/WLineEdit>
#include <Wt/WPushButton>
#include <Wt/WText>


mo::UI::wt::MainApplication::MainApplication(const WEnvironment& env)
    : WApplication(env)
{
    setTitle("EagleEye Web");                               // application title
    enableUpdates();
}
void mo::UI::wt::MainApplication::requestUpdate()
{
    _dirty = true;
    auto current_time = boost::posix_time::microsec_clock::universal_time();
    if ((current_time - _last_update_time).total_milliseconds() > 15)
    {
        this->triggerUpdate();
    }
}
#endif
