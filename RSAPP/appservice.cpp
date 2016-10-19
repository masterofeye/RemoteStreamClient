#include <QMessageBox>
#include "appservice.h"

AppService::AppService()
{
}

QStringList AppService::GetServers()
{
    QStringList servers = QStringList();
    servers << "Server1" << "Server2";
    return servers;
}

QMap<QString, QString> AppService::GetParameters()
{
    QMap<QString, QString> map;
    map["Crop X position"] = "125";
    map["Crop Y position"] = "678";
    return map;
}

void AppService::SaveParameters(QMap<QString, QString> parameters)
{
}

bool AppService::Connect(QString server)
{
    return true;
}

void AppService::Disconnect()
{
}

QStringList AppService::GetUsers()
{
    QStringList users;
    users << "User1" << "User2";
    return users;
}

void AppService::Approve(QStringList users)
{
    QMessageBox::information(0, "info", "approved", QMessageBox::Ok);
}

void AppService::Kick(QStringList users)
{
    QMessageBox::information(0, "info", "kicked", QMessageBox::Ok);
}
