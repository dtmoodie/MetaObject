#pragma once
#include <MetaObject/detail/Export.hpp>
#include <MetaObject/detail/TypeInfo.hpp>
#include <Wt/WApplication>

namespace mo
{
    class IParam;
    namespace UI
    {
        namespace wt
        {
            class IParamProxy;
            class IParamInputProxy;
            class IParamOutputProxy;
            class MO_EXPORTS MainApplication : public Wt::WApplication
            {
              public:
                MainApplication(const Wt::WEnvironment& env);
                void requestUpdate();

              private:
                void greet();
                bool _dirty;
                boost::posix_time::ptime _last_update_time;
            };
        }
    }
}
