#ifndef MO_THREAD_FIBER_INCLUDE_HPP
#define MO_THREAD_FIBER_INCLUDE_HPP

// This file dispatches includes to patched include files found in core/jetson_patch
// which contain syntax fixes for compiling on the jetson tk1 with its broken nvcc 6.5 compiler -_-
#if defined(BOOST_fiber_errorS_H)
#error "Must include this file before fiber/exceptions.hpp"
#endif

#if (defined(__NVCC__) && defined(__GNUC__) && (__GNUC__ == 4))
#ifdef _GLIBCXX_FUTURE
#error "Must include this file before <future>"
#endif

#include <MetaObject/thread/boost_patch/future>

#include <MetaObject/thread/boost_patch/exceptions.hpp>

#include <MetaObject/thread/boost_patch/spinlock_status.hpp>

#include <MetaObject/thread/boost_patch/spinlock_ttas.hpp>
#include <MetaObject/thread/boost_patch/spinlock_ttas_adaptive.hpp>
#include <MetaObject/thread/boost_patch/spinlock_ttas_adaptive_futex.hpp>
#include <MetaObject/thread/boost_patch/spinlock_ttas_futex.hpp>

#include <MetaObject/thread/boost_patch/context.hpp>

#ifdef BOOST_FIBERS_FIBER_H
#error "Must include this before boost/fiber/fiber.hpp"
#endif

#include <MetaObject/thread/boost_patch/fiber.hpp>

#include <MetaObject/thread/boost_patch/scheduler.hpp>

#include <MetaObject/thread/boost_patch/condition_variable.hpp>
#include <MetaObject/thread/boost_patch/mutex.hpp>
#include <MetaObject/thread/boost_patch/recursive_mutex.hpp>

#else // defined(__NVCC__) && defined(__GNUC__) && __GNUC__ == 4

// On all other systems, just include the boost headers since nvcc actually works
#include <boost/fiber/condition_variable.hpp>
#include <boost/fiber/context.hpp>
#include <boost/fiber/detail/spinlock.hpp>
#include <boost/fiber/fiber.hpp>
#include <boost/fiber/mutex.hpp>
#include <boost/fiber/recursive_timed_mutex.hpp>
#include <boost/fiber/scheduler.hpp>
#endif

#include <boost/fiber/detail/config.hpp>
#include <boost/fiber/properties.hpp>
#include <boost/thread/lock_types.hpp>
#endif // MO_THREAD_FIBER_INCLUDE_HPP
