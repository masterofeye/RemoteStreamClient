#pragma once
#include <QtTest/QtTest>
#include "OpenVXWrapper.h"
#include "BaseTestUnit.hpp"
#include "TestSuite.hpp"

#define NAME "GraphUnitTest"

class GraphUnitTest :public TestSuite
{
    Q_OBJECT
public:

    GraphUnitTest() : TestSuite(NAME)
    {
    }
private:

    std::shared_ptr<spdlog::logger> m_Logger;
    RW::CORE::Context *m_Context;
    RW::CORE::Graph *m_Graph;
    private slots:
    void initTestCase()
    {
        m_Logger = spdlog::stdout_logger_mt("dummy_logger2");
        m_Context = new RW::CORE::Context(m_Logger);
    }

    void init()
    {
       
    }
    void Graph_CreateContext_CreationPositive()
    {
        m_Graph = new RW::CORE::Graph(m_Context, m_Logger);
        QVERIFY2(m_Graph->IsInitialized(), "Graph coudln't initilized");
    }

    void Graph_Verification_Negative()
    {
        QVERIFY2(!m_Graph->IsGraphVerified(), "Graph is verified.");
    }

    void Graph_DoVerification_Positive()
    {
        QVERIFY2((m_Graph->VerifyGraph() == RW::tenStatus::nenSuccess), "Graph isn't verified durring execution of VerifyGraph");
    }

    void Graph_Verification_Positive()
    {
        QVERIFY2(m_Graph->IsGraphVerified(), "Graph isn't verified.");
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

static GraphUnitTest instance;