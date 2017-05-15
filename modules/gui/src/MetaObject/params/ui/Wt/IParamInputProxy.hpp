#pragma once
#include <MetaObject/params/UI/WT.hpp>
#include <Wt/WContainerWidget>
#include <Wt/WComboBox>
#include <Wt/WText>
namespace mo
{
    namespace UI
    {
        namespace wt
        {
            class IParamInputProxy : public Wt::WContainerWidget
            {
            public:
                IParamInputProxy(InputParam* param_, MainApplication* app_,
                    WContainerWidget *parent_ = 0) :
                    Wt::WContainerWidget(parent_),
                    _input_param(param_),
                    _app(app_),
                    _combo_box(new Wt::WComboBox(this)),
                    _item_text(new Wt::WText(this))

                {
                    _item_text->setText(param_->getTreeName());
                    _item_text->setToolTip(mo::Demangle::TypeToName(param_->getTypeInfo()));
                    _combo_box->changed().connect(
                        std::bind([this]()
                    {
                        if (onInputSelected)
                            onInputSelected(_combo_box->currentText());
                    }));
                    if (IParam* current_input = _input_param->getInputParam())
                    {
                        _combo_box->addItem(current_input->getTreeName());
                    }
                }
                void SetAvailableInputs(const std::vector<std::string>& inputs)
                {
                    auto lock = _app->getUpdateLock();
                    _combo_box->clear();
                    std::string current_input_name;
                    if (IParam* current_input = _input_param->getInputParam())
                    {
                        current_input_name = current_input->getTreeName();
                    }
                    for (int i = 0; i < inputs.size(); ++i)
                    {
                        _combo_box->addItem(inputs[i]);
                        if (inputs[i] == current_input_name)
                            _combo_box->setCurrentIndex(i);
                    }
                    _app->requestUpdate();
                }
                void setInputSelectionCallback(const std::function<void(const Wt::WString&)>& cb)
                {
                    onInputSelected = cb;
                }
            protected:
                InputParam* _input_param;
                MainApplication* _app;
                Wt::WComboBox* _combo_box;
                Wt::WText* _item_text;
                std::function<void(const Wt::WString&)> onInputSelected;
            };
        } // wt
    } // UI
} // mo