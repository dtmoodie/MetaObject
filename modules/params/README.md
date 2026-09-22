# params

The parameter module has a class hierarchy as follows:

```
                        IParam            (base interface: name, flags, data type, access)
        /               |                    \
  IControlParam        IPublisher           ISubscriber
        |                  |                     |
 ITControlParam<T>   TParam<IPublisher>    TParam<ISubscriber>
 (PARAM/STATE/         -> TPublisher          -> TSubscriber<T>
  STATUS macro)          |                       |
                    TFPublisher<T,            TSubscriberPtr<T>  (INPUT macro member;
                    FLAG=kOUTPUT>              specialized for std::vector<T,A>,
                    (SOURCE macro)             ct::TArrayView<const T>,
                                               std::shared_ptr<const T>)

  Multi-value variants: IMultiPublisher / TMultiOutput (TMultiPublisher.hpp),
  IMultiSubscriber / TMultiSubscriber.
```

Note: the `Input`/`Output` -> `Subscriber`/`Publisher` rename is **in progress**. `TMultiInput`/`TMultiOutput` are the pre-rename names and still exist; `TSubscriberPtr::Input_t` remains the member type used by the `INPUT` macro. When touching this module, prefer the new naming for anything new.

Other components:

- `Header` — per-update metadata carried with data parameters
- `buffers/` — `IBuffer` implementations: `Queue`, `Map`, circular (`detail/CircularBufferImpl`), `NNStreamBuffer`, `StreamBuffer`/`BlockingStreamBuffer`, created through `BufferFactory`
- `ParamServer` / `NamedParam` / `ParamAllocator` — the named-parameter registry
- `AccessToken` — access control for parameters
- `TStlAllocator` / `ParamAllocator` — memory allocation for param data
