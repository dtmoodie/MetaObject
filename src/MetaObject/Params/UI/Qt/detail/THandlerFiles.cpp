#ifdef HAVE_QT5
#include "MetaObject/Params/UI/Qt/POD.hpp"
#include "MetaObject/Params/Types.hpp"
#include <boost/thread/recursive_mutex.hpp>
#include "qfiledialog.h"
#include "qpushbutton.h"
using namespace mo;
using namespace mo::UI;
using namespace mo::UI::qt;

THandler<WriteDirectory, void>::THandler(): 
    btn(nullptr), 
    parent(nullptr), 
    _currently_updating(false)
{
}

void THandler<WriteDirectory, void>::UpdateUi( WriteDirectory* data)
{
    if(data && IHandler::getParamMtx())
    {
        mo::Mutex_t::scoped_lock lock(*IHandler::getParamMtx());
        _currently_updating = true;
        btn->setText(QString::fromStdString(data->string()));
        _currently_updating = false;
    }                    
}

void THandler<WriteDirectory, void>::onUiUpdate(QObject* sender)
{
    if(_currently_updating)
        return;
    if (sender == btn && IHandler::getParamMtx())
    {
        mo::Mutex_t::scoped_lock lock(*IHandler::getParamMtx());
        QString filename;
        
        filename = QFileDialog::getExistingDirectory(parent, "Select save directory");
        
        btn->setText(filename);
        *fileData = WriteDirectory(filename.toStdString());
        if(onUpdate)
            onUpdate();
        if(_listener)
            _listener->onUpdate(this);
    }                    
}

void THandler<WriteDirectory, void>::SetData(WriteDirectory* data_)
{
    fileData = data_;
    if (btn)
        UpdateUi(data_);
}

WriteDirectory* THandler<WriteDirectory, void>::GetData()
{
    return fileData;
}
std::vector<QWidget*> THandler<WriteDirectory, void>::GetUiWidgets(QWidget* parent_)
{

    std::vector< QWidget* > output;
    parent = parent_;
    if (btn == nullptr)
        btn = new QPushButton(parent);
    btn->connect(btn, SIGNAL(clicked()), proxy, SLOT(on_update()));
    output.push_back(btn);
    return output;
}



THandler<ReadDirectory, void>::THandler(): 
    btn(nullptr), 
    parent(nullptr), 
    _currently_updating(false)
{
}

void THandler<ReadDirectory, void>::UpdateUi( ReadDirectory* data)
{
    if(data && IHandler::getParamMtx())
    {
        mo::Mutex_t::scoped_lock lock(*IHandler::getParamMtx());
        _currently_updating = true;
        btn->setText(QString::fromStdString(data->string()));
        _currently_updating = false;
    }                    
}

void THandler<ReadDirectory, void>::onUiUpdate(QObject* sender)
{
    if(_currently_updating)
        return;
    if (sender == btn && IHandler::getParamMtx())
    {
        mo::Mutex_t::scoped_lock lock(*IHandler::getParamMtx());
        QString filename;

        filename = QFileDialog::getExistingDirectory(parent, "Select read directory");

        
        btn->setText(filename);
        *fileData = ReadDirectory(filename.toStdString());
        if(onUpdate)
            onUpdate();
        if(_listener)
            _listener->onUpdate(this);
    }                    
}

void THandler<ReadDirectory, void>::SetData(ReadDirectory* data_)
{
    fileData = data_;
    if (btn)
        UpdateUi(data_);
}

ReadDirectory* THandler<ReadDirectory, void>::GetData()
{
    return fileData;
}
std::vector<QWidget*> THandler<ReadDirectory, void>::GetUiWidgets(QWidget* parent_)
{

    std::vector< QWidget* > output;
    parent = parent_;
    if (btn == nullptr)
        btn = new QPushButton(parent);
    btn->connect(btn, SIGNAL(clicked()), proxy, SLOT(on_update()));
    output.push_back(btn);
    return output;
}




THandler<ReadFile, void>::THandler(): 
    btn(nullptr), 
    parent(nullptr), 
    _currently_updating(false)
{
}

void THandler<ReadFile, void>::UpdateUi( ReadFile* data)
{
    if(data && IHandler::getParamMtx())
    {
        mo::Mutex_t::scoped_lock lock(*IHandler::getParamMtx());
        _currently_updating = true;
        btn->setText(QString::fromStdString(data->string()));
        _currently_updating = false;
    }                    
}

void THandler<ReadFile, void>::onUiUpdate(QObject* sender)
{
    if(_currently_updating)
        return;
    if (sender == btn && IHandler::getParamMtx())
    {
        mo::Mutex_t::scoped_lock lock(*IHandler::getParamMtx());
        QString filename;

        filename = QFileDialog::getOpenFileName(parent, "Select file to open");


        btn->setText(filename);
        *fileData = ReadFile(filename.toStdString());
        if(onUpdate)
            onUpdate();
        if(_listener)
            _listener->onUpdate(this);
    }                    
}

void THandler<ReadFile, void>::SetData(ReadFile* data_)
{
    fileData = data_;
    if (btn)
        UpdateUi(data_);
}

ReadFile* THandler<ReadFile, void>::GetData()
{
    return fileData;
}
std::vector<QWidget*> THandler<ReadFile, void>::GetUiWidgets(QWidget* parent_)
{

    std::vector< QWidget* > output;
    parent = parent_;
    if (btn == nullptr)
        btn = new QPushButton(parent);
    btn->connect(btn, SIGNAL(clicked()), proxy, SLOT(on_update()));
    output.push_back(btn);
    return output;
}


THandler<Writypedefile, void>::THandler(): 
    btn(nullptr), 
    parent(nullptr), 
    _currently_updating(false)
{
}

void THandler<Writypedefile, void>::UpdateUi( Writypedefile* data)
{
    if(data && IHandler::getParamMtx())
    {
        mo::Mutex_t::scoped_lock lock(*IHandler::getParamMtx());
        _currently_updating = true;
        btn->setText(QString::fromStdString(data->string()));
        _currently_updating = false;
    }                    
}

void THandler<Writypedefile, void>::onUiUpdate(QObject* sender)
{
    if(_currently_updating)
        return;
    if (sender == btn && IHandler::getParamMtx())
    {
        mo::Mutex_t::scoped_lock lock(*IHandler::getParamMtx());
        QString filename;

        filename = QFileDialog::getSaveFileName(parent, "Select file to save");


        btn->setText(filename);
        *fileData = Writypedefile(filename.toStdString());
        if(onUpdate)
            onUpdate();
        if(_listener)
            _listener->onUpdate(this);
    }                    
}

void THandler<Writypedefile, void>::SetData(Writypedefile* data_)
{
    fileData = data_;
    if (btn)
        UpdateUi(data_);
}

Writypedefile* THandler<Writypedefile, void>::GetData()
{
    return fileData;
}
std::vector<QWidget*> THandler<Writypedefile, void>::GetUiWidgets(QWidget* parent_)
{

    std::vector< QWidget* > output;
    parent = parent_;
    if (btn == nullptr)
        btn = new QPushButton(parent);
    btn->connect(btn, SIGNAL(clicked()), proxy, SLOT(on_update()));
    output.push_back(btn);
    return output;
}

#endif
