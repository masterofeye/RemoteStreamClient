#pragma once

#include <QtTest/QtTest>
#include "OpenVXWrapper.h"
#include "BaseTestUnit.hpp"
#include "TestSuite.hpp"

#define NAME "ContextUnitTest"

class ContextUnitTest : public TestSuite
{
    Q_OBJECT
public:

    ContextUnitTest() : TestSuite(NAME)
    {
    }
private: 
    RW::CORE::Context *m_Context;
    std::shared_ptr<spdlog::logger> m_Logger;
    private slots:
        void initTestCase()
        {
            m_Logger = spdlog::stdout_logger_mt("dummy_logger1");
        }

        void init()
        {
        }

        void Context_CreateContext_CreationPositive()
        {
            m_Context = new RW::CORE::Context(m_Logger);
            QVERIFY2(m_Context->IsInitialized(), "Context coudln't initilized.");
        }

        void Context_GetVendor_Positiv()
        {
            QVERIFY2(m_Context->Vendor() != 0, "Vendor number wrong.");
        }

        void Context_GetVersion_Positiv()
        {
            QVERIFY2(m_Context->Version() != 0, "Version number wrong.");
        }

        void cleanup()
        {

        }


        void cleanupTestCase()
        {
            delete m_Context;
            m_Logger.reset();
        }
};

static ContextUnitTest instance;