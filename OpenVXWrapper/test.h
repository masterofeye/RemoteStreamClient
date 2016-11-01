#ifndef TEST_H
#define TEST_H

#include <QObject>

class Test : public QObject
{
    Q_OBJECT

public:
    Test(QObject *parent);
    ~Test();

private:
    
};

#endif // TEST_H
