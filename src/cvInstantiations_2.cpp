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
#include "parameters/UI/Qt/OpenCV.hpp"
#include "parameters/Parameters.hpp"
#include "parameters/Types.hpp"
#include <vector>
#include <string>
#include <type_traits>

using namespace Parameters;

#ifdef HAVE_OPENCV
#include <opencv2/core.hpp>

static TypedParameter<cv::Point> point_("instance");
static TypedParameter<cv::Point2d> point2d_("instance");
static TypedParameter<cv::Point2f> point2f_("instance");

static TypedParameter<cv::Vec2d> vec2d_("instance");
static TypedParameter<cv::Vec2f> vec2f_("instance");
static TypedParameter<cv::Vec2i> vec2i_("instance");

static TypedParameter<cv::Vec3d> vec3d_("instance");
static TypedParameter<cv::Vec3f> vec3f_("instance");
static TypedParameter<cv::Vec3i> vec3i_("instance");
static TypedParameter<cv::Range> range_("instance");

static TypedParameter<cv::Rect_<int>> recti("instance");
static TypedParameter<cv::Rect_<double>> rectd("instance");
static TypedParameter<cv::Rect_<float>> rectf("instance");

CEREAL_REGISTER_TYPE(TypedParameter<cv::Point>);
CEREAL_REGISTER_TYPE(TypedParameter<cv::Point2d>);
CEREAL_REGISTER_TYPE(TypedParameter<cv::Point2f>);
CEREAL_REGISTER_TYPE(TypedParameter<cv::Vec2d>);
CEREAL_REGISTER_TYPE(TypedParameter<cv::Vec2f>);
CEREAL_REGISTER_TYPE(TypedParameter<cv::Vec2i>);
CEREAL_REGISTER_TYPE(TypedParameter<cv::Vec3d>);
CEREAL_REGISTER_TYPE(TypedParameter<cv::Vec3f>);
CEREAL_REGISTER_TYPE(TypedParameter<cv::Vec3i>);

#endif