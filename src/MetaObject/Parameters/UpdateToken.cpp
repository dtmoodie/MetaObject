#include "MetaObject/IMetaObject.hpp"

using namespace mo;

UpdateToken::UpdateToken(IParameter& param)
{

}

UpdateToken::~UpdateToken()
{
    if(_cs)
        _param.SetCoordinateSystem(_cs);
    _param.Commit(_ts, _ctx, _fn);
}

UpdateToken& UpdateToken::operator()(time_t&& ts)
{
    _ts = ts;
}

UpdateToken& UpdateToken::operator()(size_t fn)
{
    _fn = fn;
}

UpdateToken& UpdateToken::operator()(Context* ctx)
{
    _ctx = ctx;
}

UpdateToken& UpdateToken::operator()(ICoordinateSystem* cs)
{
    _cs = cs;
}
