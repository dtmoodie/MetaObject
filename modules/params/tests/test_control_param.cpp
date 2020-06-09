#include <MetaObject/params/TControlParam.hpp>

#include <MetaObject/serialization/BinaryLoader.hpp>
#include <MetaObject/serialization/BinarySaver.hpp>
#include <MetaObject/serialization/JSONPrinter.hpp>

#include <gtest/gtest.h>

#include "../../runtime_reflection/tests/common.hpp"

struct control_param : testing::Test
{
    control_param()
    {
        param.setName("param");
        param.setValue(5.0F);
    }
    mo::TControlParam<float> param;
};

TEST_F(control_param, print)
{
    std::stringstream ss;
    param.print(ss);
    std::string str = ss.str();
    ASSERT_EQ(str, "[param] float = 5");
}

TEST_F(control_param, visit_static)
{
    std::stringstream ss;
    mo::PrintVisitor visitor(ss);
    param.visit(visitor);
    ASSERT_EQ(ss.str(),
              "name: std::string\n    name: "
              "char\nroot_name: std::string\n    "
              "root_name: char\nflags: mo::ParamFlags\n  data: float\n");
}

TEST_F(control_param, save_json)
{
    std::stringstream ss;

    {
        mo::JSONSaver saver(ss);
        param.save(saver);
    }
    ASSERT_EQ(ss.str(),
              "{\n    \"name\": \"param\",\n    \"root_name\": \"\",\n    \"flags\": {\n        \"value\": "
              "\"kCONTROL\"\n    },\n    \"data\": 5.0\n}");
}

TEST_F(control_param, load_json)
{
    std::stringstream ss;

    {
        mo::JSONSaver saver(ss);
        param.save(saver);
    }
    mo::JSONLoader loader(ss);
    param.setValue(10.0F);
    ASSERT_EQ(param.getValue(), 10.0F);
    param.load(loader);
    ASSERT_EQ(param.getValue(), 5.0F);
}

TEST_F(control_param, load_save_binary)
{
    std::stringstream ss;

    {
        mo::BinarySaver saver(ss);
        param.save(saver);
    }
    mo::BinaryLoader loader(ss);
    param.setValue(10.0F);
    ASSERT_EQ(param.getValue(), 10.0F);
    param.load(loader);
    ASSERT_EQ(param.getValue(), 5.0F);
}

struct control_param_ptr : testing::Test
{
    control_param_ptr()
    {
        param.setName("param");
        param.setUserDataPtr(&val);
        param.setValue(5.0F);
    }
    mo::TControlParam<float*> param;
    float val;
};

TEST_F(control_param_ptr, print)
{
    std::stringstream ss;
    param.print(ss);
    std::string str = ss.str();
    ASSERT_EQ(str, "[param] float = 5");
}

TEST_F(control_param_ptr, visit_static)
{
    std::stringstream ss;
    mo::PrintVisitor visitor(ss);
    param.visit(visitor);
    ASSERT_EQ(ss.str(),
              "name: std::string\n    name: "
              "char\nroot_name: std::string\n    "
              "root_name: char\nflags: mo::ParamFlags\n  data: float\n");
}

TEST_F(control_param_ptr, save_json)
{
    std::stringstream ss;

    {
        mo::JSONSaver saver(ss);
        param.save(saver);
    }
    ASSERT_EQ(ss.str(),
              "{\n    \"name\": \"param\",\n    \"root_name\": \"\",\n    \"flags\": {\n        \"value\": "
              "\"kCONTROL\"\n    },\n    \"data\": 5.0\n}");
}

TEST_F(control_param_ptr, load_json)
{
    std::stringstream ss;

    {
        mo::JSONSaver saver(ss);
        param.save(saver);
    }
    mo::JSONLoader loader(ss);
    param.setValue(10.0F);
    ASSERT_EQ(param.getValue(), 10.0F);
    ASSERT_EQ(val, 10.0F);
    param.load(loader);
    ASSERT_EQ(param.getValue(), 5.0F);
    ASSERT_EQ(val, 5.0F);
}

TEST_F(control_param_ptr, load_save_binary)
{
    std::stringstream ss;

    {
        mo::BinarySaver saver(ss);
        param.save(saver);
    }
    mo::BinaryLoader loader(ss);
    param.setValue(10.0F);
    ASSERT_EQ(param.getValue(), 10.0F);
    ASSERT_EQ(val, 10.0F);
    param.load(loader);
    ASSERT_EQ(val, 5.0F);
}