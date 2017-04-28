#ifdef HAVE_WT
#include <MetaObject/Params/UI/Wt/Enum.hpp>
using namespace mo::UI::wt;
using namespace mo;

TParamProxy<EnumParam, void>::TParamProxy(ITParam<EnumParam>* param_,
    MainApplication* app_,
    WContainerWidget* parent_) :
    IParamProxy(param_, app_, parent_),
    _param(param_)
{
    _combo_box = new Wt::WComboBox(this);
    for (auto& name : param_->GetDataPtr()->enumerations)
    {
        _combo_box->addItem(name);
    }
    _combo_box->changed().connect(std::bind(&TParamProxy<EnumParam, void>::onUiChanged, this));
}



void TParamProxy<EnumParam, void>::SetTooltip(const std::string& tip)
{
    auto lock = _app->getUpdateLock();
    _combo_box->setToolTip(tip);
    _app->requestUpdate();
}

void TParamProxy<EnumParam, void>::onParamUpdate(mo::Context* ctx, mo::IParam* param)
{
    auto lock = _app->getUpdateLock();
    _combo_box->clear();
    mo::Mutex_t::scoped_lock param_lock(param->mtx());
    for (auto& name : _param->GetDataPtr()->enumerations)
    {
        _combo_box->addItem(name);
    }
    _app->requestUpdate();
}

void TParamProxy<EnumParam, void>::onUiChanged()
{
    mo::Mutex_t::scoped_lock lock(_param->mtx());
    std::vector<std::string>& enums = _param->GetDataPtr()->enumerations;
    for (int i = 0; i < enums.size(); ++i)
    {
        if (enums[i] == _combo_box->currentText())
        {
            _param->GetDataPtr()->currentSelection = i;
            return;
        }
    }
}

#endif

