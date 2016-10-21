#include "mainwindow.hpp"
#include "ui_mainwindow.h"
#include <QLineEdit>
#include <QtWidgets/QWidget>
#include "..\OpenVXWrapperTest\Pipeline.hpp"

#include <QDebug>


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    m_mwUI(new Ui::MainWindow)
{
    m_isConnected = false;
    m_isOperator = false;
    m_mwUI->setupUi(this);
    QStringList headers;
    headers << "Parameter" << "Value";
    m_mwUI->treeConfig->setHeaderLabels(headers);
    m_mwUI->serversList->addItems(m_appService.GetServers());
    m_mwUI->operatorTabs->hide();
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
        m_mwUI->treeConfig->setItemWidget(treeItem, 1, new QLineEdit(iparam.value(), m_mwUI->treeConfig));
    }

    m_mwUI->treeConfig->addTopLevelItem(rootItem);
    rootItem->setExpanded(true);
    m_mwUI->treeConfig->resizeColumnToContents(0);
}

void MainWindow::FillUsers()
{
    foreach(const QString &userName, m_appService.GetUsers())
    {
        auto listItem = new QListWidgetItem(0, m_mwUI->usersList);
        m_mwUI->usersList->addItem(listItem);
        m_mwUI->usersList->setItemWidget(listItem, new QCheckBox(userName));
    }
}

MainWindow::~MainWindow()
{
    delete m_mwUI;
}

void MainWindow::on_connectButton_clicked()
{
    if (m_isConnected)
    {
        // Stop playing...
        UpdateUI();

        // Disconnect
        m_appService.Disconnect();
        m_isConnected = false;
    }
    else
    {
        auto server = m_mwUI->serversList->currentItem()->text();
        m_isOperator = m_mwUI->operatorCheckBox->isChecked();
        m_isConnected = m_appService.Connect(server);

        UpdateUI();

        if (m_isConnected)
        {
            if (m_isOperator)
            {
                FillParameters();
                FillUsers();
            }
            else
            {
                auto file_logger = spdlog::stdout_logger_mt("file_logger");
                file_logger->debug("******************");
                file_logger->debug("*Applicationstart*");
                file_logger->debug("******************");

                tstPipelineParams pipeparams;
				pipeparams.file_logger = file_logger;
				pipeparams.pViewer = m_pViewer;

				CPipeline pipe(&pipeparams);
                CPipethread thread;

                QObject::connect(&thread, SIGNAL(started()), &pipe, SLOT(RunPipeline()), Qt::DirectConnection);
                pipe.moveToThread(&thread);

                thread.start();
            }
        }
    }
}

void MainWindow::UpdateUI()
{
    m_mwUI->connectButton->setText(m_isConnected ? "Disconnect" : "Connect");
    m_mwUI->operatorCheckBox->setEnabled(!m_isConnected && m_mwUI->serversList->currentRow() >= 0);
    m_mwUI->serversList->setEnabled(!m_isConnected);
    m_mwUI->operatorTabs->setEnabled(m_isConnected);
    m_mwUI->usersTab->setEnabled(m_isConnected);
    m_mwUI->labelNoConnection->setVisible(!m_isConnected);

    if (m_isConnected)
        UpdateTabVisibility();
    else
        DisconnectAndClear();
}


void MainWindow::UpdateTabVisibility()
{
    if (m_isOperator)
    {
        m_mwUI->operatorTabs->show();
        m_mwUI->usersTab->hide();
    }
    else
    {
        m_mwUI->operatorTabs->hide();

        m_pViewer = new RW::VPL::QT_SIMPLE::VPL_Viewer();

        m_pViewer->setParams(1920, 720);
        //pViewer->setParams(640, 480);
    QImage::Format format;
#ifdef DEC_INTEL
    format = QImage::Format::Format_RGBX8888;
#endif
#ifdef DEC_NVENC
    format = QImage::Format::Format_RGB888;
#endif
    m_pViewer->setImgType(format);

        m_mwUI->usersTab->show();
        m_mwUI->usersTab->removeTab(0);
        m_mwUI->usersTab->addTab(m_pViewer, "Video");
    }
}

void MainWindow::DisconnectAndClear()
{
    m_mwUI->operatorTabs->hide();

    m_mwUI->usersTab->show();
    m_mwUI->usersTab->removeTab(0);
    m_mwUI->usersTab->addTab(m_mwUI->videoTab, "Video");

    m_mwUI->treeConfig->clear();
    m_mwUI->usersList->clear();
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
    for (int i = 0; i < m_mwUI->usersList->count(); i++)
    {
        QCheckBox * checkBox = (QCheckBox *)m_mwUI->usersList->itemWidget(m_mwUI->usersList->item(i));
        checkBox->setChecked(checked);
    }
}

QStringList MainWindow::GetSelectedUsers()
{
    QStringList result;
    for (int i = 0; i < m_mwUI->usersList->count(); i++)
    {
        QCheckBox * checkBox = (QCheckBox *)m_mwUI->usersList->itemWidget(m_mwUI->usersList->item(i));
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
    QTreeWidgetItemIterator it(m_mwUI->treeConfig);
    it++;
    while(*it)
    {
        auto name = (*it)->text(0);
        auto value = ((QLineEdit *)m_mwUI->treeConfig->itemWidget(*it, 1))->text();
        parameters[name] = value;
        ++it;
    }

    m_appService.SaveParameters(parameters);
}

void MainWindow::on_serversList_itemClicked(QListWidgetItem *item)
{
    m_mwUI->connectButton->setEnabled(true);
}
