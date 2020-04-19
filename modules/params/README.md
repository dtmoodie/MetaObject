The parameter module has a class hierarchy as follows:


// Access tokens?
                    IParam
    IPublisher             ISubscriber
    TParam<IPublisher>     TParam<ISubscriber>
    TPublisher             TSubscriber
                           TSubscriberPtr (specialization of TSubscriber?)