#include "MetaObject/types/file_types.hpp"
#include "MetaObject/params/ui/Qt/POD.hpp"
#include "qlineedit.h"
#include <boost/thread/recursive_mutex.hpp>

using namespace mo;
using namespace mo::UI;
using namespace mo::UI::qt;

THandler<std::string, void>::THandler(IParamProxy& parent) : UiUpdateHandler(parent), lineEdit(nullptr)
{
}

void THandler<std::string, void>::updateUi(const std::string& data)
{
    if (lineEdit)
        lineEdit->setText(QString::fromStdString(data));
}

void THandler<std::string, void>::updateParam(std::string& data)
{
    if (lineEdit)
        data = lineEdit->text().toStdString();
}

std::vector<QWidget*> THandler<std::string, void>::getUiWidgets(QWidget* parent)
{
    std::vector<QWidget*> output;
    _parent_widget = parent;
    if (lineEdit == nullptr)
        lineEdit = new QLineEdit(parent);
    lineEdit->connect(lineEdit, SIGNAL(returnPressed()), proxy, SLOT(on_update()));
    output.push_back(lineEdit);
    return output;
}
