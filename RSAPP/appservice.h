#ifndef APPSERVICE_H
#define APPSERVICE_H
#include <QStringList>
#include <QMap>

class AppService
{
public:
    AppService();

    QStringList GetServers();

    QMap<QString, QString> GetParameters();

    void SaveParameters(QMap<QString, QString> parameters);

    bool Connect(QString server);

    void Disconnect();

    QStringList GetUsers();

    void Approve(QStringList users);

    void Kick(QStringList users);
};

#endif // APPSERVICE_H
