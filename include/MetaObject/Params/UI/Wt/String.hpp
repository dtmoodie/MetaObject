#pragma once

#include "IParamProxy.hpp"
#include <MetaObject/Params/ITParam.hpp>
#include <Wt/WLineEdit>

namespace mo
{
namespace UI
{
namespace wt
{
    template<>
    class MO_EXPORTS TDataProxy<std::string, void>
    {
    public:
        static const int IS_DEFAULT = false;
        TDataProxy();
        void CreateUi(IParamProxy* proxy, std::string* data, bool read_only);
        void UpdateUi(const std::string& data);
        void onUiUpdate(std::string& data);
        void SetTooltip(const std::string& tp);
    protected:
        Wt::WLineEdit* _line_edit = nullptr;
    };

} // namespace wt
} // namespace UI
} // namespace mo
