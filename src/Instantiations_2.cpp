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

#include "parameters/Parameters.hpp"
#include "parameters/Types.hpp"
#include <vector>
#include <string>
#include <type_traits>

using namespace Parameters;

static TypedParameter<char> char_("instance");
static TypedParameter<unsigned char> uchar_("instance");
static TypedParameter<short> short_("instance");
static TypedParameter<unsigned short> ushort_("instance");
static TypedParameter<int> int_("instance");
static TypedParameter<unsigned int> uint_("instance");
static TypedParameter<long> long_("instance");
static TypedParameter<unsigned long> ulong_("instance");
static TypedParameter<long long> longlong_("instance");

CEREAL_REGISTER_TYPE(TypedParameter<char>);
CEREAL_REGISTER_TYPE(TypedParameter<unsigned char>);
CEREAL_REGISTER_TYPE(TypedParameter<short>);
CEREAL_REGISTER_TYPE(TypedParameter<unsigned short>);
CEREAL_REGISTER_TYPE(TypedParameter<int>);
CEREAL_REGISTER_TYPE(TypedParameter<unsigned int>);
CEREAL_REGISTER_TYPE(TypedParameter<long>);
CEREAL_REGISTER_TYPE(TypedParameter<unsigned long>);
CEREAL_REGISTER_TYPE(TypedParameter<long long>);

/*static TypedParameter<unsigned long long> ulonglong_("instance");
static TypedParameter<float> float_("instance");
static TypedParameter<double> double_("instance");


static Parameters::TypedParameter<std::string> string_("instance");
static Parameters::TypedParameter<std::vector<std::string>> vecString_("instance");
static Parameters::TypedParameter<Parameters::ReadFile> readFile_("instance");
static Parameters::TypedParameter<Parameters::WriteFile> writeFile_("instance");
static Parameters::TypedParameter<Parameters::ReadDirectory> readDir_("instance");
static Parameters::TypedParameter<Parameters::WriteDirectory> writeDir_("instance");
static Parameters::TypedParameter<Parameters::EnumParameter> enum_("instance");
*/
