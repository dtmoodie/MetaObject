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



static TypedParameter<std::vector<char>> vecchar_("instance");
static TypedParameter<std::vector<unsigned char>> vecuchar_("instance");
static TypedParameter<std::vector<short>> vecshort_("instance");
static TypedParameter<std::vector<unsigned short>> vecushort_("instance");
static TypedParameter<std::vector<int>> vecint_("instance");
static TypedParameter<std::vector<unsigned int>> vecuint_("instance");
static TypedParameter<std::vector<long>> veclong_("instance");
static TypedParameter<std::vector<unsigned long>> veculong_("instance");
static TypedParameter<std::vector<long long>> veclonglong_("instance");
static TypedParameter<std::vector<unsigned long long>> veculonglong_("instance");
static TypedParameter<std::vector<float>> vecfloat_("instance");
static TypedParameter<std::vector<double>> vecdouble_("instance");
