#include "mainwindow.h"
#include "MetaObject/params/TParamPtr.hpp"
#include "MetaObject/params/detail/TParamPtrImpl.hpp"
#include "MetaObject/params/ui/Qt/IParamProxy.hpp"
#include "MetaObject/params/ui/Qt/TParamProxy.hpp"
#include "MetaObject/params/ui/WidgetFactory.hpp"
#include "ui_mainwindow.h"

#include <MetaObject/runtime_reflection/visitor_traits/vector.hpp>
#include <MetaObject/runtime_reflection/visitor_traits/string.hpp>
#include <MetaObject/runtime_reflection/visitor_traits/pair.hpp>

#include <MetaObject/thread/fiber_include.hpp>
// The following lines are commented out to demonstrate user interface instantiation in a different translation unit
// Since the instantiation library is included, instantiations of several types are registered with the full user
// interface code for those types.  Thus the following are not needed for those types.  However, not all types are
// included, so a few of the Params will use the default met Param
//#include "MetaObject/params/ui/Qt/POD.hpp"
#ifdef HAVE_OPENCV
#include "MetaObject/types/opencv.hpp"
//#include "MetaObject/params/ui/Qt/OpenCV.hpp"
#endif
//#include "MetaObject/params/ui/Qt/Containers.hpp"

#include "MetaObject/params/ITParam.hpp"
#include "MetaObject/params/TParamPtr.hpp"
//#include "MetaObject/params/RangedParam.hpp"
#include <MetaObject/MetaParameters.hpp>

#include <cereal/types/string.hpp>
#include <cereal/types/utility.hpp>
#include <cereal/types/vector.hpp>

using namespace mo;
MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    m_system_table = SystemTable::instance();
    initMetaParamsModule(m_system_table.get());
    ui->setupUi(this);
    {
        /*auto param = new mo::RangedParam<std::vector<float>>(0.0,20,"vector float");
        param->GetDataPtr()->push_back(15.0);
        param->GetDataPtr()->push_back(14.0);
        param->GetDataPtr()->push_back(13.0);
        param->GetDataPtr()->push_back(12.0);
        param->GetDataPtr()->push_back(11.0);
        Params.push_back(std::shared_ptr<IParam>(param));*/
    } {
        auto param = new mo::TParam<std::vector<int>>("vector int");
        auto token = param->access();
        token().push_back(15);
        token().push_back(14);
        token().push_back(13);
        token().push_back(12);
        token().push_back(11);
        Params.push_back(std::shared_ptr<IParam>(param));
    }
    {
#ifdef HAVE_OPENCV
        auto param = new mo::TParam<cv::Point3f>("Point3f");
        auto token = param->access();
        token().z = 15;
        Params.push_back(std::shared_ptr<IParam>(param));
#endif
    }
    {
        auto param =
            new TParam<std::vector<std::pair<std::string, std::string>>>("Vector std::pair<std::string, std::string>");
        auto token = param->access();
        token().push_back(std::pair<std::string, std::string>("asdf", "1234"));
        Params.push_back(std::shared_ptr<IParam>(param));
    }
    {
#ifdef HAVE_OPENCV
        std::shared_ptr<IParam> param(new TParamPtr<std::vector<cv::Point2f>>("Vector cv::Point2f", &testRefVec));
        testRefVec.push_back(cv::Point2f(0, 1));
        testRefVec.push_back(cv::Point2f(2, 3));
        testRefVec.push_back(cv::Point2f(4, 5));
        testRefVec.push_back(cv::Point2f(6, 7));
        testRefVec.push_back(cv::Point2f(8, 1));
        testRefVec.push_back(cv::Point2f(9, 1));
        testRefVec.push_back(cv::Point2f(10, 1));
        param->emitUpdate();
        Params.push_back(param);
    }
    {

        std::shared_ptr<IParam> param(new mo::TParamPtr<std::vector<cv::Scalar>>("Vector cv::Scalar", &testRefScalar));
        testRefScalar.push_back(cv::Scalar(0));
        testRefScalar.push_back(cv::Scalar(1));
        testRefScalar.push_back(cv::Scalar(2));
        testRefScalar.push_back(cv::Scalar(3));
        testRefScalar.push_back(cv::Scalar(4));
        testRefScalar.push_back(cv::Scalar(5));
        testRefScalar.push_back(cv::Scalar(6));
        testRefScalar.push_back(cv::Scalar::all(7));
        testRefScalar.push_back(cv::Scalar(8));
        param->emitUpdate();
        Params.push_back(param);
#endif
    }
    {
        auto param = new TParam<int>("int");
        auto token = param->access();
        token() = 10;
        Params.push_back(std::shared_ptr<IParam>(param));
    }
#ifdef HAVE_OPENCV
    {
        auto param = new TParam<cv::Scalar>("scalar");
        Params.push_back(std::shared_ptr<IParam>(param));
    }
    {

        auto param = new TParam<cv::Matx<double, 4, 4>>("Mat4x4d");
        Params.push_back(std::shared_ptr<IParam>(param));
    }
    {
        auto param = new TParam<cv::Vec<double, 6>>("Vec6d");
        Params.push_back(std::shared_ptr<IParam>(param));
    }
#endif
    for (int i = 0; i < Params.size(); ++i)
    {
        auto proxy = mo::UI::qt::WidgetFactory::Instance()->CreateProxy(Params[i].get());
        ui->widgetLayout->addWidget(proxy->getParamWidget(this));
        proxies.push_back(proxy);
    }
}

MainWindow::~MainWindow()
{
    delete ui;
}
void MainWindow::on_btnSerialize_clicked()
{
    /*{
        cv::FileStorage fs("test.yml", cv::FileStorage::WRITE);
        for (int i = 0; i < Params.size(); ++i)
        {
            Params::Persistence::cv::Serialize(&fs, Params[i].get());
        }
    }
    {
        cv::FileStorage fs("test.yml", cv::FileStorage::READ);
        cv::FileNode node = fs.root();
        std::cout << node.name().c_str() << std::endl;
        for (int i = 0; i < Params.size(); ++i)
        {
            Params::Persistence::cv::DeSerialize(&node, Params[i].get());
        }
    }*/
}
