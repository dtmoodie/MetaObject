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

https://github.com/dtmoodie/MetaObject
*/
#ifndef MO_PARAMS_ISUBSCRIBER_HPP
#define MO_PARAMS_ISUBSCRIBER_HPP

#include <MetaObject/detail/Export.hpp>
#include <MetaObject/params/IParam.hpp>

#include <memory>
namespace mo
{

    struct MO_EXPORTS ISubscriber : virtual IParam
    {
        ISubscriber();
        virtual ~ISubscriber();

        // This gets a pointer to the Publisher that feeds into this input
        virtual IPublisher* getPublisher() const = 0;

        virtual bool setInput(std::shared_ptr<IPublisher> param) = 0;
        virtual bool setInput(IPublisher* param = nullptr) = 0;
        virtual bool acceptsPublisher(const IPublisher& param) const = 0;
        virtual bool acceptsType(const TypeInfo& type) const = 0;
        virtual std::vector<TypeInfo> getInputTypes() const = 0;
        virtual bool isInputSet() const = 0;

        // These methods can be used for querying the state of the subscriber
        // getInputData returns the next data available from the publisher
        // that this subscriber is subscribing to
        // If stream = nullptr, do not perform any automatic synchronization in this method call
        // virtual IDataContainerConstPtr_t getInputData(IAsyncStream* = nullptr) const = 0;

        // This method returns the current data element
        // If stream = nullptr, do not perform any automatic synchronization in this method call
        virtual IDataContainerConstPtr_t getCurrentData(IAsyncStream* = nullptr) const = 0;

        virtual bool hasNewData() const = 0;
        virtual boost::optional<Header> getNewestHeader() const = 0;
        virtual OptionalTime getNewestTimestamp() const;

        // Request data at the desired header
        // Will set currentData to  the data provided, thus further calls to getCurrentData will return
        // the result of this call
        // If desired == nullptr, return the newest data element
        // the stream parameter will be used to perform synchronization, if stream == nullptr, attempt to sync with the
        // stream from getStream()
        virtual IDataContainerConstPtr_t getData(const Header* desired = nullptr, IAsyncStream* stream = nullptr) = 0;

        // Return the headers for all available data samples
        virtual std::vector<Header> getAvailableHeaders() const = 0;
    };
} // namespace mo
#endif // MO_PARAMS_ISUBSCRIBER_HPP