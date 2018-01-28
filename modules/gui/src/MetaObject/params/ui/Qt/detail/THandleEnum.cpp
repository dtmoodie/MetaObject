#ifdef HAVE_QT5
#include "MetaObject/types/file_types.hpp"
#include "MetaObject/params/ui/Qt/POD.hpp"
#include <boost/thread/recursive_mutex.hpp>
using namespace mo;
using namespace mo::UI;
using namespace mo::UI::qt;

THandler<EnumParam, void>::THandler(IParamProxy& parent) : UiUpdateHandler(parent), enumCombo(nullptr)
{
}

void THandler<EnumParam, void>::updateUi(const EnumParam& data)
{
    if (enumCombo)
    {
        enumCombo->clear();
        for (int i = 0; i < data.enumerations.size(); ++i)
        {
            enumCombo->addItem(QString::fromStdString(data.enumerations[i]));
        }
        enumCombo->setCurrentIndex(static_cast<int>(data.current_selection));
    }
}

void THandler<EnumParam, void>::updateParam(EnumParam& data)
{
    if (enumCombo)
    {
        data.current_selection = enumCombo->currentIndex();
    }
}

std::vector<QWidget*> THandler<EnumParam, void>::getUiWidgets(QWidget* parent)
{

    std::vector<QWidget*> output;
    if (enumCombo == nullptr)
        enumCombo = new QComboBox(parent);
    enumCombo->connect(enumCombo, SIGNAL(currentIndexChanged(int)), proxy, SLOT(on_update(int)));
    output.push_back(enumCombo);
    return output;
}
#endif
