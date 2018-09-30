#include "JSONPrinter.hpp"

namespace mo
{
    JSONWriter::JSONWriter(std::ostream& os) : m_ar(os) {}

    template <class T>
    IWriteVisitor& JSONWriter::writePod(const T* ptr, const std::string& name, const size_t cnt)
    {
        if (!name.empty())
        {
            m_ar.setNextName(name.c_str());
        }
        else
        {
        }
        for (size_t i = 0; i < cnt; ++i)
        {
            m_ar(ptr[i]);
        }
        return *this;
    }

    IWriteVisitor& JSONWriter::operator()(const char* val, const std::string& name, const size_t cnt)
    {
        return writePod(val, name, cnt);
    }

    IWriteVisitor& JSONWriter::operator()(const int8_t* val, const std::string& name, const size_t cnt)
    {
        return writePod(val, name, cnt);
    }

    IWriteVisitor& JSONWriter::operator()(const uint8_t* val, const std::string& name, const size_t cnt)
    {
        return writePod(val, name, cnt);
    }

    IWriteVisitor& JSONWriter::operator()(const int16_t* val, const std::string& name, const size_t cnt)
    {
        return writePod(val, name, cnt);
    }

    IWriteVisitor& JSONWriter::operator()(const uint16_t* val, const std::string& name, const size_t cnt)
    {
        return writePod(val, name, cnt);
    }

    IWriteVisitor& JSONWriter::operator()(const int32_t* val, const std::string& name, const size_t cnt)
    {
        return writePod(val, name, cnt);
    }

    IWriteVisitor& JSONWriter::operator()(const uint32_t* val, const std::string& name, const size_t cnt)
    {
        return writePod(val, name, cnt);
    }

    IWriteVisitor& JSONWriter::operator()(const int64_t* val, const std::string& name, const size_t cnt)
    {
        return writePod(val, name, cnt);
    }

    IWriteVisitor& JSONWriter::operator()(const uint64_t* val, const std::string& name, const size_t cnt)
    {
        return writePod(val, name, cnt);
    }

    IWriteVisitor& JSONWriter::operator()(const float* val, const std::string& name, const size_t cnt)
    {
        return writePod(val, name, cnt);
    }

    IWriteVisitor& JSONWriter::operator()(const double* val, const std::string& name, const size_t cnt)
    {
        return writePod(val, name, cnt);
    }

    IWriteVisitor& JSONWriter::operator()(const void* val, const std::string& name, const size_t cnt)
    {
        m_ar.saveBinaryValue(val, cnt, name.c_str());
        return *this;
    }

    IWriteVisitor& JSONWriter::operator()(const IStructTraits* val, const std::string& name)
    {
        if (!name.empty())
        {
            m_ar.setNextName(name.c_str());
        }
        m_ar.startNode();
        WriteCache::operator()(val, name);
        m_ar.finishNode();
        return *this;
    }

    IWriteVisitor& JSONWriter::operator()(const IContainerTraits* val, const std::string& name)
    {
        if (!name.empty())
        {
            m_ar.setNextName(name.c_str());
        }
        if (val->type() != TypeInfo(typeid(std::string)))
        {
            m_ar.startNode();
            if (val->keyType() != TypeInfo(typeid(std::string)))
            {
                m_ar.makeArray();
            }
        }
        val->visit(this);
        if (val->type() != TypeInfo(typeid(std::string)))
        {
            m_ar.finishNode();
        }
        return *this;
    }

    VisitorTraits JSONWriter::traits() const
    {
        VisitorTraits out;
        out.supports_named_access = true;
        out.reader = false;
        return out;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///                                      JSONReader
    /////////////////////////////////////////////////////////////////////////////////////////////////////////

    JSONReader::JSONReader(std::istream& os) : m_ar(os) {}

    template <class T>
    IReadVisitor& JSONReader::readPod(T* ptr, const std::string& name, const size_t cnt)
    {
        if (!name.empty())
        {
            m_ar.setNext(name.c_str(), false);
        }
        for (size_t i = 0; i < cnt; ++i)
        {
            m_ar(ptr[i]);
            auto name = m_ar.getNodeName();
            if (name)
            {
                m_last_read_name = name;
            }
        }
        return *this;
    }

    IReadVisitor& JSONReader::operator()(char* val, const std::string& name, const size_t cnt)
    {
        return readPod(val, name, cnt);
    }
    IReadVisitor& JSONReader::operator()(int8_t* val, const std::string& name, const size_t cnt)
    {
        return readPod(val, name, cnt);
    }
    IReadVisitor& JSONReader::operator()(uint8_t* val, const std::string& name, const size_t cnt)
    {
        return readPod(val, name, cnt);
    }
    IReadVisitor& JSONReader::operator()(int16_t* val, const std::string& name, const size_t cnt)
    {
        return readPod(val, name, cnt);
    }
    IReadVisitor& JSONReader::operator()(uint16_t* val, const std::string& name, const size_t cnt)
    {
        return readPod(val, name, cnt);
    }
    IReadVisitor& JSONReader::operator()(int32_t* val, const std::string& name, const size_t cnt)
    {
        return readPod(val, name, cnt);
    }
    IReadVisitor& JSONReader::operator()(uint32_t* val, const std::string& name, const size_t cnt)
    {
        return readPod(val, name, cnt);
    }
    IReadVisitor& JSONReader::operator()(int64_t* val, const std::string& name, const size_t cnt)
    {
        return readPod(val, name, cnt);
    }
    IReadVisitor& JSONReader::operator()(uint64_t* val, const std::string& name, const size_t cnt)
    {
        return readPod(val, name, cnt);
    }

    IReadVisitor& JSONReader::operator()(float* val, const std::string& name, const size_t cnt)
    {
        return readPod(val, name, cnt);
    }
    IReadVisitor& JSONReader::operator()(double* val, const std::string& name, const size_t cnt)
    {
        return readPod(val, name, cnt);
    }
    IReadVisitor& JSONReader::operator()(void* val, const std::string& name, const size_t cnt)
    {
        m_ar.loadBinaryValue(val, cnt, name.c_str());
        return *this;
    }

    IReadVisitor& JSONReader::operator()(IStructTraits* val, const std::string& name)
    {
        if (!name.empty())
        {
            m_ar.setNext(name.c_str(), false);
        }
        auto name_ptr = m_ar.getNodeName();
        m_ar.startNode();

        ReadCache::operator()(val, name);
        if (name_ptr)
        {
            m_last_read_name = name_ptr;
        }

        m_ar.finishNode();
        return *this;
    }

    IReadVisitor& JSONReader::operator()(IContainerTraits* val, const std::string& name)
    {
        if (!name.empty())
        {
            m_ar.setNext(name.c_str(), false);
        }
        if (val->type() != TypeInfo(typeid(std::string)))
        {
            auto parent_name = m_ar.getNodeName();
            m_ar.startNode();
            if (val->keyType() != TypeInfo(typeid(std::string)))
            {
                uint64_t size;
                m_ar.loadSize(size);
                val->setSize(size);
            }
            else
            {
                uint64_t count = 0;
                while (true)
                {
                    const auto name = m_ar.getNodeName();
                    if (!name)
                    {
                        break;
                    }
                    count += 1;
                    m_ar.startNode();
                    m_ar.finishNode();
                }
                val->setSize(count);
                m_ar.finishNode();
                m_ar.setNext(parent_name, false);
                m_ar.startNode();
            }
        }
        val->visit(this);
        if (val->type() != TypeInfo(typeid(std::string)))
        {
            m_ar.finishNode();
        }
        return *this;
    }

    VisitorTraits JSONReader::traits() const
    {
        VisitorTraits out;
        out.supports_named_access = true;
        out.reader = true;
        return out;
    }

    std::string JSONReader::getCurrentElementName() const { return m_last_read_name; }
}
