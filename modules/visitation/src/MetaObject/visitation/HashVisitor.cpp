#include "HashVisitor.hpp"
#include <ct/Hash.hpp>

namespace mo
{

size_t HashVisitor::generateObjecthash(const IStructTraits *traits)
{
    m_hash = 0;
    traits->visit(this);
    return m_hash;
}


VisitorTraits HashVisitor::traits() const
{
    VisitorTraits output;
    output.reader = true;
    output.supports_named_access = true;
    return output;
}


template<class T>
void HashVisitor::hash(const std::string& name)
{
    std::hash<std::string> name_hasher;
    m_hash = ct::combineHash<size_t>(m_hash, name_hasher(name));
    m_hash = ct::combineHash<size_t>(m_hash, name_hasher(mo::TypeInfo(typeid(T)).name()));
}

IWriteVisitor& HashVisitor::operator()(const bool* val, const std::string& name, const size_t cnt)
{
    hash<bool>(name);
    return *this;
}

IWriteVisitor& HashVisitor::operator()(const char* val, const std::string& name, const size_t cnt)
{
    hash<char>(name);
    return *this;
}

IWriteVisitor& HashVisitor::operator()(const int8_t* val, const std::string& name, const size_t cnt)
{
    hash<int8_t>(name);
    return *this;
}

IWriteVisitor& HashVisitor::operator()(const uint8_t* val, const std::string& name, const size_t cnt)
{
    hash<uint8_t>(name);
    return *this;
}

IWriteVisitor& HashVisitor::operator()(const int16_t* val, const std::string& name, const size_t cnt)
{
    hash<int16_t>(name);
    return *this;
}

IWriteVisitor& HashVisitor::operator()(const uint16_t* val, const std::string& name, const size_t cnt)
{
    hash<uint16_t>(name);
    return *this;
}

IWriteVisitor& HashVisitor::operator()(const int32_t* val, const std::string& name, const size_t cnt)
{
    hash<int32_t>(name);
    return *this;
}

IWriteVisitor& HashVisitor::operator()(const uint32_t* val, const std::string& name, const size_t cnt)
{
    hash<uint32_t>(name);
    return *this;
}

IWriteVisitor& HashVisitor::operator()(const int64_t* val, const std::string& name, const size_t cnt)
{
    hash<int64_t>(name);
    return *this;
}

IWriteVisitor& HashVisitor::operator()(const uint64_t* val, const std::string& name, const size_t cnt)
{

    hash<uint64_t>(name);
    return *this;
}

IWriteVisitor& HashVisitor::operator()(const long long* val, const std::string& name, const size_t cnt)
{

    hash<long long>(name);
    return *this;
}

IWriteVisitor& HashVisitor::operator()(const unsigned long long* val, const std::string& name, const size_t cnt)
{

    hash<unsigned long long>(name);
    return *this;
}

IWriteVisitor& HashVisitor::operator()(const float* val, const std::string& name, const size_t cnt)
{
    hash<float>(name);
    return *this;
}

IWriteVisitor& HashVisitor::operator()(const double* val, const std::string& name, const size_t cnt)
{
    hash<double>(name);
    return *this;
}

IWriteVisitor& HashVisitor::operator()(const void* binary, const std::string& name, const size_t bytes)
{
    hash<void>(name);
    return *this;
}


IWriteVisitor& HashVisitor::operator()(const IStructTraits* val, const std::string& name)
{
    m_hash = ct::combineHash<size_t>(m_hash, std::hash<std::string>{}(name));
    val->visit(this);
    return *this;
}

IWriteVisitor& HashVisitor::operator()(const IContainerTraits* val, const std::string& name)
{
    m_hash = ct::combineHash<size_t>(m_hash, std::hash<std::string>{}(name));
    val->visit(this);
    return *this;
}


const void* HashVisitor::getPointer(const TypeInfo type, const uint64_t id)
{
    return nullptr;
}

void HashVisitor::setSerializedPointer(const TypeInfo type, const uint64_t id, const void* ptr)
{

}

std::unique_ptr<CacheDataContainer>& HashVisitor::accessCache(const std::string& name, const uint64_t id)
{
    return m_ptr;
}

}
