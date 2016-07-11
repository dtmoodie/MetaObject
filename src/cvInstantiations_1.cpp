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
#include <opencv2/core/cuda.hpp>
static TypedParameter<std::vector<cv::Point>> vecpoint_("instance");
static TypedParameter<std::vector<cv::Point2d>> vecpoint2d_("instance");
static TypedParameter<std::vector<cv::Point2f>> vecpoint2f_("instance");
//static TypedParameter<std::vector<cv::Vec2b>> vecvec2b_("instance");
static TypedParameter<std::vector<cv::Vec2d>> vecvec2d_("instance");
static TypedParameter<std::vector<cv::Vec2f>> vecvec2f_("instance");
static TypedParameter<std::vector<cv::Vec2i>> vecvec2i_("instance");
static TypedParameter<std::vector<cv::Vec3b>> vecvec3b_("instance");
static TypedParameter<std::vector<cv::Vec3d>> vecvec3d_("instance");
static TypedParameter<std::vector<cv::Vec3f>> vecvec3f_("instance");
static TypedParameter<std::vector<cv::Vec3i>> vecvec3i_("instance");

static TypedParameter<cv::cuda::GpuMat> gpumat_("instance");
static TypedParameter<cv::cuda::HostMem> hostmem_("instance");
static TypedParameter<cv::Mat> mat_("instance");


CEREAL_REGISTER_TYPE(TypedParameter<std::vector<cv::Point>>);
CEREAL_REGISTER_TYPE(TypedParameter<std::vector<cv::Point2d>>);
CEREAL_REGISTER_TYPE(TypedParameter<std::vector<cv::Point2f>>);
CEREAL_REGISTER_TYPE(TypedParameter<std::vector<cv::Vec2d>>);
CEREAL_REGISTER_TYPE(TypedParameter<std::vector<cv::Vec2i>>);
CEREAL_REGISTER_TYPE(TypedParameter<std::vector<cv::Vec3b>>);
CEREAL_REGISTER_TYPE(TypedParameter<std::vector<cv::Vec3d>>);
CEREAL_REGISTER_TYPE(TypedParameter<std::vector<cv::Vec3f>>);
CEREAL_REGISTER_TYPE(TypedParameter<std::vector<cv::Vec3i>>);
CEREAL_REGISTER_TYPE(TypedParameter<cv::Mat>);
CEREAL_REGISTER_TYPE(TypedParameter<cv::cuda::HostMem>);
CEREAL_REGISTER_TYPE(TypedParameter<cv::cuda::GpuMat>);
#endif