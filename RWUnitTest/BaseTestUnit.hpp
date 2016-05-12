#pragma once
#include <QtTest/QtTest>

class BaseTestUnit :QObject
{
    Q_OBJECT
public:
    QString Name;
    BaseTestUnit(QString Name);
    ~BaseTestUnit();
};

