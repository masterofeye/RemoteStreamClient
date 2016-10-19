/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 5.6.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QDockWidget>
#include <QtWidgets/QGraphicsView>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QListWidget>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QTreeWidget>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QWidget *centralWidget;
    QGridLayout *gridLayout_2;
    QVBoxLayout *verticalLayout_3;
    QTabWidget *operatorTabs;
    QWidget *tabConfig;
    QVBoxLayout *verticalLayout_5;
    QTreeWidget *treeConfig;
    QHBoxLayout *horizontalLayout;
    QSpacerItem *horizontalSpacer;
    QPushButton *saveButton;
    QWidget *tabStats;
    QGridLayout *gridLayout_3;
    QVBoxLayout *verticalLayout_8;
    QHBoxLayout *horizontalLayout_3;
    QVBoxLayout *verticalLayout_6;
    QHBoxLayout *horizontalLayout_2;
    QSpacerItem *horizontalSpacer_6;
    QCheckBox *checkAllBox;
    QSpacerItem *horizontalSpacer_5;
    QPushButton *approveButton;
    QPushButton *kickButton;
    QSpacerItem *horizontalSpacer_2;
    QListWidget *usersList;
    QSpacerItem *horizontalSpacer_4;
    QVBoxLayout *verticalLayout_7;
    QGridLayout *gridLayout;
    QLabel *label_2;
    QSpacerItem *horizontalSpacer_3;
    QGraphicsView *graphicsView;
    QWidget *tabVideo;
    QTabWidget *usersTab;
    QWidget *videoTab;
    QVBoxLayout *verticalLayout_4;
    QLabel *labelNoConnection;
    QDockWidget *dockWidget;
    QWidget *dockWidgetContents;
    QVBoxLayout *verticalLayout_2;
    QVBoxLayout *verticalLayout;
    QLabel *label;
    QListWidget *serversList;
    QPushButton *connectButton;
    QCheckBox *operatorCheckBox;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QStringLiteral("MainWindow"));
        MainWindow->setWindowModality(Qt::NonModal);
        MainWindow->resize(1067, 903);
        MainWindow->setFocusPolicy(Qt::TabFocus);
        centralWidget = new QWidget(MainWindow);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        gridLayout_2 = new QGridLayout(centralWidget);
        gridLayout_2->setSpacing(6);
        gridLayout_2->setContentsMargins(11, 11, 11, 11);
        gridLayout_2->setObjectName(QStringLiteral("gridLayout_2"));
        verticalLayout_3 = new QVBoxLayout();
        verticalLayout_3->setSpacing(6);
        verticalLayout_3->setObjectName(QStringLiteral("verticalLayout_3"));
        operatorTabs = new QTabWidget(centralWidget);
        operatorTabs->setObjectName(QStringLiteral("operatorTabs"));
        operatorTabs->setEnabled(false);
        operatorTabs->setDocumentMode(false);
        operatorTabs->setTabsClosable(false);
        operatorTabs->setMovable(false);
        tabConfig = new QWidget();
        tabConfig->setObjectName(QStringLiteral("tabConfig"));
        verticalLayout_5 = new QVBoxLayout(tabConfig);
        verticalLayout_5->setSpacing(6);
        verticalLayout_5->setContentsMargins(11, 11, 11, 11);
        verticalLayout_5->setObjectName(QStringLiteral("verticalLayout_5"));
        treeConfig = new QTreeWidget(tabConfig);
        QTreeWidgetItem *__qtreewidgetitem = new QTreeWidgetItem();
        __qtreewidgetitem->setText(1, QStringLiteral("2"));
        __qtreewidgetitem->setText(0, QStringLiteral("1"));
        treeConfig->setHeaderItem(__qtreewidgetitem);
        treeConfig->setObjectName(QStringLiteral("treeConfig"));
        treeConfig->setAutoFillBackground(false);
        treeConfig->setColumnCount(2);

        verticalLayout_5->addWidget(treeConfig);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setSpacing(6);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer);

        saveButton = new QPushButton(tabConfig);
        saveButton->setObjectName(QStringLiteral("saveButton"));

        horizontalLayout->addWidget(saveButton);


        verticalLayout_5->addLayout(horizontalLayout);

        operatorTabs->addTab(tabConfig, QString());
        tabStats = new QWidget();
        tabStats->setObjectName(QStringLiteral("tabStats"));
        gridLayout_3 = new QGridLayout(tabStats);
        gridLayout_3->setSpacing(6);
        gridLayout_3->setContentsMargins(11, 11, 11, 11);
        gridLayout_3->setObjectName(QStringLiteral("gridLayout_3"));
        verticalLayout_8 = new QVBoxLayout();
        verticalLayout_8->setSpacing(6);
        verticalLayout_8->setObjectName(QStringLiteral("verticalLayout_8"));
        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setSpacing(6);
        horizontalLayout_3->setObjectName(QStringLiteral("horizontalLayout_3"));
        verticalLayout_6 = new QVBoxLayout();
        verticalLayout_6->setSpacing(6);
        verticalLayout_6->setObjectName(QStringLiteral("verticalLayout_6"));
        verticalLayout_6->setSizeConstraint(QLayout::SetMinimumSize);
        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setSpacing(6);
        horizontalLayout_2->setObjectName(QStringLiteral("horizontalLayout_2"));
        horizontalLayout_2->setSizeConstraint(QLayout::SetFixedSize);
        horizontalSpacer_6 = new QSpacerItem(3, 20, QSizePolicy::Fixed, QSizePolicy::Minimum);

        horizontalLayout_2->addItem(horizontalSpacer_6);

        checkAllBox = new QCheckBox(tabStats);
        checkAllBox->setObjectName(QStringLiteral("checkAllBox"));

        horizontalLayout_2->addWidget(checkAllBox);

        horizontalSpacer_5 = new QSpacerItem(80, 20, QSizePolicy::Fixed, QSizePolicy::Minimum);

        horizontalLayout_2->addItem(horizontalSpacer_5);

        approveButton = new QPushButton(tabStats);
        approveButton->setObjectName(QStringLiteral("approveButton"));
        QFont font;
        font.setUnderline(true);
        approveButton->setFont(font);
        approveButton->setStyleSheet(QStringLiteral("color: rgb(48, 12, 255);"));
        approveButton->setFlat(true);

        horizontalLayout_2->addWidget(approveButton);

        kickButton = new QPushButton(tabStats);
        kickButton->setObjectName(QStringLiteral("kickButton"));
        kickButton->setFont(font);
        kickButton->setCursor(QCursor(Qt::PointingHandCursor));
        kickButton->setStyleSheet(QStringLiteral("color: rgb(48, 12, 255);"));
        kickButton->setFlat(true);

        horizontalLayout_2->addWidget(kickButton);

        horizontalSpacer_2 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_2->addItem(horizontalSpacer_2);


        verticalLayout_6->addLayout(horizontalLayout_2);

        usersList = new QListWidget(tabStats);
        usersList->setObjectName(QStringLiteral("usersList"));
        QSizePolicy sizePolicy(QSizePolicy::Fixed, QSizePolicy::Expanding);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(usersList->sizePolicy().hasHeightForWidth());
        usersList->setSizePolicy(sizePolicy);
        usersList->setStyleSheet(QLatin1String("\n"
"                                        QListView::item {\n"
"                                        border: 0px;\n"
"                                        padding-left: 2px;\n"
"                                        }\n"
"                                      "));
        usersList->setEditTriggers(QAbstractItemView::NoEditTriggers);
        usersList->setSelectionMode(QAbstractItemView::NoSelection);
        usersList->setViewMode(QListView::ListMode);
        usersList->setSelectionRectVisible(false);

        verticalLayout_6->addWidget(usersList);


        horizontalLayout_3->addLayout(verticalLayout_6);

        horizontalSpacer_4 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_3->addItem(horizontalSpacer_4);


        verticalLayout_8->addLayout(horizontalLayout_3);

        verticalLayout_7 = new QVBoxLayout();
        verticalLayout_7->setSpacing(6);
        verticalLayout_7->setObjectName(QStringLiteral("verticalLayout_7"));
        gridLayout = new QGridLayout();
        gridLayout->setSpacing(6);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        label_2 = new QLabel(tabStats);
        label_2->setObjectName(QStringLiteral("label_2"));

        gridLayout->addWidget(label_2, 0, 0, 1, 1);

        horizontalSpacer_3 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout->addItem(horizontalSpacer_3, 0, 1, 1, 1);


        verticalLayout_7->addLayout(gridLayout);

        graphicsView = new QGraphicsView(tabStats);
        graphicsView->setObjectName(QStringLiteral("graphicsView"));

        verticalLayout_7->addWidget(graphicsView);


        verticalLayout_8->addLayout(verticalLayout_7);


        gridLayout_3->addLayout(verticalLayout_8, 0, 0, 1, 1);

        operatorTabs->addTab(tabStats, QString());
        tabVideo = new QWidget();
        tabVideo->setObjectName(QStringLiteral("tabVideo"));
        operatorTabs->addTab(tabVideo, QString());

        verticalLayout_3->addWidget(operatorTabs);

        usersTab = new QTabWidget(centralWidget);
        usersTab->setObjectName(QStringLiteral("usersTab"));
        videoTab = new QWidget();
        videoTab->setObjectName(QStringLiteral("tab_2"));
        verticalLayout_4 = new QVBoxLayout(videoTab);
        verticalLayout_4->setSpacing(6);
        verticalLayout_4->setContentsMargins(11, 11, 11, 11);
        verticalLayout_4->setObjectName(QStringLiteral("verticalLayout_4"));
        labelNoConnection = new QLabel(videoTab);
        labelNoConnection->setObjectName(QStringLiteral("labelNoConnection"));
        labelNoConnection->setStyleSheet(QStringLiteral("color: rgb(195, 192, 198);"));
        labelNoConnection->setAlignment(Qt::AlignCenter);

        verticalLayout_4->addWidget(labelNoConnection);

        usersTab->addTab(videoTab, QString());

        verticalLayout_3->addWidget(usersTab);


        gridLayout_2->addLayout(verticalLayout_3, 0, 0, 1, 1);

        MainWindow->setCentralWidget(centralWidget);
        dockWidget = new QDockWidget(MainWindow);
        dockWidget->setObjectName(QStringLiteral("dockWidget"));
        dockWidget->setMinimumSize(QSize(180, 213));
        dockWidget->setFeatures(QDockWidget::DockWidgetFloatable);
        dockWidget->setAllowedAreas(Qt::LeftDockWidgetArea);
        dockWidgetContents = new QWidget();
        dockWidgetContents->setObjectName(QStringLiteral("dockWidgetContents"));
        verticalLayout_2 = new QVBoxLayout(dockWidgetContents);
        verticalLayout_2->setSpacing(6);
        verticalLayout_2->setContentsMargins(11, 11, 11, 11);
        verticalLayout_2->setObjectName(QStringLiteral("verticalLayout_2"));
        verticalLayout = new QVBoxLayout();
        verticalLayout->setSpacing(6);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        label = new QLabel(dockWidgetContents);
        label->setObjectName(QStringLiteral("label"));

        verticalLayout->addWidget(label);

        serversList = new QListWidget(dockWidgetContents);
        serversList->setObjectName(QStringLiteral("serversList"));

        verticalLayout->addWidget(serversList);

        connectButton = new QPushButton(dockWidgetContents);
        connectButton->setObjectName(QStringLiteral("connectButton"));
        connectButton->setEnabled(false);

        verticalLayout->addWidget(connectButton);

        operatorCheckBox = new QCheckBox(dockWidgetContents);
        operatorCheckBox->setObjectName(QStringLiteral("operatorCheckBox"));
        operatorCheckBox->setEnabled(false);

        verticalLayout->addWidget(operatorCheckBox);


        verticalLayout_2->addLayout(verticalLayout);

        dockWidget->setWidget(dockWidgetContents);
        MainWindow->addDockWidget(static_cast<Qt::DockWidgetArea>(1), dockWidget);

        retranslateUi(MainWindow);

        operatorTabs->setCurrentIndex(0);
        usersTab->setCurrentIndex(0);


        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "Remote Streaming App SSH", 0));
        saveButton->setText(QApplication::translate("MainWindow", "Save", 0));
        operatorTabs->setTabText(operatorTabs->indexOf(tabConfig), QApplication::translate("MainWindow", "Configuration", 0));
        checkAllBox->setText(QString());
        approveButton->setText(QApplication::translate("MainWindow", "Approve", 0));
        kickButton->setText(QApplication::translate("MainWindow", "Kick", 0));
        label_2->setText(QApplication::translate("MainWindow", "Bandwidth usage", 0));
        operatorTabs->setTabText(operatorTabs->indexOf(tabStats), QApplication::translate("MainWindow", "Statistics", 0));
        operatorTabs->setTabText(operatorTabs->indexOf(tabVideo), QApplication::translate("MainWindow", "Video", 0));
        labelNoConnection->setText(QApplication::translate("MainWindow", "Connection is not established", 0));
        usersTab->setTabText(usersTab->indexOf(videoTab), QApplication::translate("MainWindow", "Video", 0));
        dockWidget->setWindowTitle(QApplication::translate("MainWindow", "Connection", 0));
        label->setText(QApplication::translate("MainWindow", "Remote Servers:", 0));
        connectButton->setText(QString());
        operatorCheckBox->setText(QApplication::translate("MainWindow", "as Operator", 0));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
