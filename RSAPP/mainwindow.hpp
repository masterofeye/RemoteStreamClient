#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QListWidgetItem>
#include "appservice.h"

#include "..\VPL\QT_simple\VPL_Viewer.hpp"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private slots:

    void on_connectButton_clicked();

    void on_serversList_currentRowChanged(int currentRow);

    void on_usersList_itemClicked(QListWidgetItem *item);

    void on_checkAllBox_toggled(bool checked);

    void on_approveButton_clicked();

    void on_kickButton_clicked();

    void on_saveButton_clicked();

    void on_serversList_itemClicked(QListWidgetItem *item);

private:
    Ui::MainWindow *m_mwUI;
	RW::VPL::QT_SIMPLE::VPL_Viewer *m_pViewer;
    AppService m_appService;
    bool m_isConnected;
    bool m_isOperator;

    void FillParameters();
    void FillUsers();
    void UpdateUI();
    void Disconnect();
    void DisconnectAndClear();
    void UpdateTabVisibility();
    QStringList GetSelectedUsers();
};

#endif // MAINWINDOW_H
