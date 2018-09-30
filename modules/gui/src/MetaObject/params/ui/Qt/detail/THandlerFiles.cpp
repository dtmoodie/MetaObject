#include "MetaObject/types/file_types.hpp"
#include "MetaObject/params/ui/Qt/POD.hpp"
#include "qfiledialog.h"
#include "qpushbutton.h"
#include <boost/thread/recursive_mutex.hpp>
using namespace mo;
using namespace mo::UI;
using namespace mo::UI::qt;

THandler<WriteDirectory, void>::THandler(IParamProxy& parent) : UiUpdateHandler(parent), btn(nullptr)
{
}

void THandler<WriteDirectory, void>::updateUi(const WriteDirectory& data)
{
    if (btn)
    {
        btn->setText(QString::fromStdString(data.string()));
    }
}

void THandler<WriteDirectory, void>::updateParam(WriteDirectory& data)
{
    if (btn)
    {
        QString filename;
        filename = QFileDialog::getExistingDirectory(_parent_widget, "Select save directory");
        btn->setText(filename);
        data = WriteDirectory(filename.toStdString());
    }
}

std::vector<QWidget*> THandler<WriteDirectory, void>::getUiWidgets(QWidget* parent_)
{
    std::vector<QWidget*> output;
    _parent_widget = parent_;
    if (btn == nullptr)
        btn = new QPushButton(parent);
    btn->connect(btn, SIGNAL(clicked()), proxy, SLOT(on_update()));
    output.push_back(btn);
    return output;
}

THandler<ReadDirectory, void>::THandler(IParamProxy& parent) : UiUpdateHandler(parent), btn(nullptr)
{
}

void THandler<ReadDirectory, void>::updateUi(const ReadDirectory& data)
{
    if (btn)
    {
        btn->setText(QString::fromStdString(data.string()));
    }
}

void THandler<ReadDirectory, void>::updateParam(ReadDirectory& data)
{
    QString filename = QFileDialog::getExistingDirectory(parent, "Select read directory");
    data = ReadDirectory(filename.toStdString());
}

std::vector<QWidget*> THandler<ReadDirectory, void>::getUiWidgets(QWidget* parent_)
{
    std::vector<QWidget*> output;
    _parent_widget = parent_;
    parent = parent_;
    if (btn == nullptr)
        btn = new QPushButton(parent);
    btn->connect(btn, SIGNAL(clicked()), proxy, SLOT(on_update()));
    output.push_back(btn);
    return output;
}

THandler<ReadFile, void>::THandler(IParamProxy& parent) : UiUpdateHandler(parent), btn(nullptr)
{
}

void THandler<ReadFile, void>::updateUi(const ReadFile& data)
{
    _updating = true;
    if (btn)
        btn->setText(QString::fromStdString(data.string()));
    _updating = false;
}

void THandler<ReadFile, void>::updateParam(ReadFile& data)
{
    auto filename = QFileDialog::getOpenFileName(_parent_widget, "Select file to open");
    data = ReadFile(filename.toStdString());
}

std::vector<QWidget*> THandler<ReadFile, void>::getUiWidgets(QWidget* parent_)
{
    std::vector<QWidget*> output;
    _parent_widget = parent_;
    if (btn == nullptr)
        btn = new QPushButton(parent_);
    btn->connect(btn, SIGNAL(clicked()), proxy, SLOT(on_update()));
    output.push_back(btn);
    return output;
}

THandler<WriteFile, void>::THandler(IParamProxy& parent) : UiUpdateHandler(parent), btn(nullptr)
{
}

void THandler<WriteFile, void>::updateUi(const WriteFile& data)
{
    if (btn)
        btn->setText(QString::fromStdString(data.string()));
}

void THandler<WriteFile, void>::updateParam(WriteFile& data)
{
    auto filename = QFileDialog::getSaveFileName(_parent_widget, "Select file to save");
    data = WriteFile(filename.toStdString());
}

std::vector<QWidget*> THandler<WriteFile, void>::getUiWidgets(QWidget* parent_)
{
    std::vector<QWidget*> output;
    _parent_widget = parent_;
    if (btn == nullptr)
        btn = new QPushButton(parent_);
    btn->connect(btn, SIGNAL(clicked()), proxy, SLOT(on_update()));
    output.push_back(btn);
    return output;
}
