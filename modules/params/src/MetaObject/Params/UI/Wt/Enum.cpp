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
    mo::ParamTraits<EnumParam>::Storage_t data;
    if(param_->getData(data)){
        for(auto& name : data->enumerations){
            _combo_box->addItem(name);
        }
    }
    _combo_box->changed().connect(std::bind(&TParamProxy<EnumParam, void>::onUiChanged, this));
}



void TParamProxy<EnumParam, void>::SetTooltip(const std::string& tip){
    auto lock = _app->getUpdateLock();
    _combo_box->setToolTip(tip);
    _app->requestUpdate();
}

void TParamProxy<EnumParam, void>::onParamUpdate(mo::Context* ctx, mo::IParam* param){
    auto lock = _app->getUpdateLock();
    _combo_box->clear();
    mo::Mutex_t::scoped_lock param_lock(param->mtx());
    mo::ParamTraits<EnumParam>::Storage_t data;
    if(_param->getData(data)){
        for (auto& name : data->enumerations){
            _combo_box->addItem(name);
        }
    }
    _app->requestUpdate();
}

void TParamProxy<EnumParam, void>::onUiChanged()
{
    mo::Mutex_t::scoped_lock lock(_param->mtx());
    mo::ParamTraits<mo::EnumParam>::Storage_t data;
    if(_param->getData(data)){
        std::vector<std::string>& enums = data->enumerations;
        for (int i = 0; i < enums.size(); ++i){
            if (enums[i] == _combo_box->currentText()){
                data->currentSelection = i;
                return;
            }
        }
    }
}

#endif

