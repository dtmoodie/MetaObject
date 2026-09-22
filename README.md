# MetaObject

MetaObject is a reflection library that sits on top of Runtime Compiled C++ (RCC).

MetaObject provides the following features:

1) Static and dynamic introspection of object parameters

2) Static and dynamic introspection of object signals / slots

3) Serialization of object state (cereal-based)

4) Runtime re-compilation and hot-swapping of object classes (via `MetaObjectFactory`)

5) Plugin loading of compiled object classes (`.so`/`.dll`)

6) Python bindings (Boost.Python, requires OpenCV + numpy) and Qt/Wt parameter editors

All objects created with the appropriate macros and inheritance from `IMetaObject` are runtime modifiable classes. The user-facing macro API (inside `MO_DERIVE` ... `MO_END`) is:

- `PARAM(type, name, init)` / `ENUM_PARAM` / `STATE` / `STATUS` — runtime control parameters
- `INPUT(type, name)` / `OUTPUT(type, name)` — data parameters; these create `TSubscriber`/`TPublisher` members (the `Input`/`Output` naming is being migrated to `Subscriber`/`Publisher`, as are the multi-value variants `TMultiSubscriber`/`TMultiPublisher`)
- `MO_SIGNAL` / `MO_SLOT` / `PARAM_UPDATE_SLOT` — signals and slots

A quick example program using MetaObject can be found in the `examples/console_reflection` folder; `examples/console_recompile` demonstrates the runtime recompile loop.

Build/test instructions, the module layout, and current development constraints live in [CLAUDE.md](CLAUDE.md). Tests are per-module googletest binaries (`test_mo_core`, `test_mo_object`, `test_mo_params`, ...) built into the build dir.
