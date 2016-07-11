/*
Copyright (c) 2015 Daniel Moodie.
All rights reserved.

Redistribution and use in source and binary forms are permitted
provided that the above copyright notice and this paragraph are
duplicated in all such forms and that any documentation,
advertising materials, and other materials related to such
distribution and use acknowledge that the software was developed
by the Daniel Moodie. The name of
Daniel Moodie may not be used to endorse or promote products derived
from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.

https://github.com/dtmoodie/parameters
*/
#ifndef PARAMTERS_USE_UI
#define PARAMTERS_USE_UI
#endif

#include "parameters/UI/Qt/POD.hpp"
#include "parameters/Parameters.hpp"
#include "parameters/Types.hpp"
#include "parameters/RangedParameter.hpp"
#include <vector>
#include <string>
#include <type_traits>

using namespace Parameters;

static TypedParameter<unsigned long long> ulonglong_("instance");
static TypedParameter<float> float_("instance");
static TypedParameter<double> double_("instance");


static Parameters::TypedParameter<std::string> string_("instance");
static Parameters::TypedParameter<std::vector<std::string>> vecString_("instance");
static Parameters::TypedParameter<Parameters::ReadFile> readFile_("instance");
static Parameters::TypedParameter<Parameters::WriteFile> writeFile_("instance");
static Parameters::TypedParameter<Parameters::ReadDirectory> readDir_("instance");
static Parameters::TypedParameter<Parameters::WriteDirectory> writeDir_("instance");
static Parameters::TypedParameter<Parameters::EnumParameter> enum_("instance");

CEREAL_REGISTER_TYPE(TypedParameter<unsigned long long>);
CEREAL_REGISTER_TYPE(TypedParameter<float>);
CEREAL_REGISTER_TYPE(TypedParameter<double>);
CEREAL_REGISTER_TYPE(TypedParameter<std::string>);
CEREAL_REGISTER_TYPE(TypedParameter<std::vector<std::string>>);
CEREAL_REGISTER_TYPE(TypedParameter<Parameters::ReadFile>);
CEREAL_REGISTER_TYPE(TypedParameter<Parameters::WriteFile>);
CEREAL_REGISTER_TYPE(TypedParameter<Parameters::ReadDirectory>);
CEREAL_REGISTER_TYPE(TypedParameter<Parameters::WriteDirectory>);
CEREAL_REGISTER_TYPE(TypedParameter<Parameters::EnumParameter>);
