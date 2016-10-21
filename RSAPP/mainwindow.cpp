#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QLineEdit>
#include <QtWidgets/QWidget>

#include <QDebug>


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    m_isConnected = false;
    m_isOperator = false;
    ui->setupUi(this);
    QStringList headers;
    headers << "Parameter" << "Value";
    ui->treeConfig->setHeaderLabels(headers);
    ui->serversList->addItems(m_appService.GetServers());
    ui->operatorTabs->hide();
    UpdateUI();
}

void MainWindow::FillParameters()
{
    auto rootItem = new QTreeWidgetItem();
    rootItem->setText(0, "Parameters");

    QMapIterator<QString, QString> iparam(m_appService.GetParameters());
    while(iparam.hasNext())
    {
        iparam.next();
        auto treeItem = new QTreeWidgetItem();
        treeItem->setText(0, iparam.key());
        rootItem->addChild(treeItem);
        ui->treeConfig->setItemWidget(treeItem, 1, new QLineEdit(iparam.value(), ui->treeConfig));
    }

    ui->treeConfig->addTopLevelItem(rootItem);
    rootItem->setExpanded(true);
    ui->treeConfig->resizeColumnToContents(0);
}

void MainWindow::FillUsers()
{
    foreach(const QString &userName, m_appService.GetUsers())
    {
        auto listItem = new QListWidgetItem(0, ui->usersList);
        ui->usersList->addItem(listItem);
        ui->usersList->setItemWidget(listItem, new QCheckBox(userName));
    }
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_connectButton_clicked()
{
    if (m_isConnected)
    {
        // Stop playing...

        // Disconnect
        m_appService.Disconnect();
        m_isConnected = false;
    }
    else
    {
        auto server = ui->serversList->currentItem()->text();
        m_isOperator = ui->operatorCheckBox->isChecked();
        m_isConnected = m_appService.Connect(server);
        if (m_isConnected)
        {
            if (m_isOperator)
            {
                FillParameters();
                FillUsers();
            }
            else
            {
                // Start playing...
            }
        }
    }

    UpdateUI();
}

void MainWindow::UpdateUI()
{
    ui->connectButton->setText(m_isConnected ? "Disconnect" : "Connect");
    ui->operatorCheckBox->setEnabled(!m_isConnected && ui->serversList->currentRow() >= 0);
    ui->serversList->setEnabled(!m_isConnected);
    ui->operatorTabs->setEnabled(m_isConnected);
    ui->usersTab->setEnabled(m_isConnected);
    ui->labelNoConnection->setVisible(!m_isConnected);

    if (m_isConnected)
        UpdateTabVisibility();
    else
        DisconnectAndClear();
}


void MainWindow::UpdateTabVisibility()
{
    if (m_isOperator)
    {
        ui->operatorTabs->show();
        ui->usersTab->hide();
    }
    else
    {
        ui->operatorTabs->hide();

        pViewer = new RW::VPL::QT_SIMPLE::VPL_Viewer();
        ui->usersTab->show();
        ui->usersTab->removeTab(0);
        ui->usersTab->addTab(pViewer, "Video");
    }
}

void MainWindow::DisconnectAndClear()
{
    ui->operatorTabs->hide();

    ui->usersTab->show();
    ui->usersTab->removeTab(0);
    ui->usersTab->addTab(ui->videoTab, "Video");

    ui->treeConfig->clear();
    ui->usersList->clear();
}

void MainWindow::on_serversList_currentRowChanged(int currentRow)
{
    //UpdateUI();
}

void MainWindow::on_usersList_itemClicked(QListWidgetItem *item)
{
    if (item->checkState() == Qt::Checked)
    {
        item->setCheckState(Qt::Unchecked);
    }
    else
    {
        item->setCheckState(Qt::Checked);
    }
}

void MainWindow::on_checkAllBox_toggled(bool checked)
{
    for(int i=0; i < ui->usersList->count(); i++)
    {
        QCheckBox * checkBox = (QCheckBox *)ui->usersList->itemWidget(ui->usersList->item(i));
        checkBox->setChecked(checked);
    }
}

QStringList MainWindow::GetSelectedUsers()
{
    QStringList result;
    for(int i=0; i < ui->usersList->count(); i++)
    {
        QCheckBox * checkBox = (QCheckBox *)ui->usersList->itemWidget(ui->usersList->item(i));
        if (checkBox->isChecked())
        {
            result << checkBox->text();
        }
    }

    return result;
}

void MainWindow::on_approveButton_clicked()
{
    auto selectedUsers = GetSelectedUsers();
    if (selectedUsers.count() > 0)
    {
        m_appService.Approve(selectedUsers);
    }
}

void MainWindow::on_kickButton_clicked()
{
    auto selectedUsers = GetSelectedUsers();
    if (selectedUsers.count() > 0)
    {
        m_appService.Kick(selectedUsers);
    }
}

void MainWindow::on_saveButton_clicked()
{
    QMap<QString, QString> parameters;
    QTreeWidgetItemIterator it(ui->treeConfig);
    it++;
    while(*it)
    {
        auto name = (*it)->text(0);
        auto value = ((QLineEdit *)ui->treeConfig->itemWidget(*it, 1))->text();
        parameters[name] = value;
        ++it;
    }

    m_appService.SaveParameters(parameters);
}

void MainWindow::on_serversList_itemClicked(QListWidgetItem *item)
{
    ui->connectButton->setEnabled(true);
}
