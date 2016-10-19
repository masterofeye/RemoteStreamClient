/********************************************************************************
** Form generated from reading UI file 'operatorwindow.ui'
**
** Created by: Qt User Interface Compiler version 5.6.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_OPERATORWINDOW_H
#define UI_OPERATORWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QDialog>
#include <QtWidgets/QDialogButtonBox>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_OperatorWindow
{
public:
    QDialogButtonBox *buttonBox;
    QTabWidget *tabWidget;
    QWidget *tab;
    QWidget *tab_2;

    void setupUi(QDialog *OperatorWindow)
    {
        if (OperatorWindow->objectName().isEmpty())
            OperatorWindow->setObjectName(QStringLiteral("OperatorWindow"));
        OperatorWindow->resize(400, 300);
        buttonBox = new QDialogButtonBox(OperatorWindow);
        buttonBox->setObjectName(QStringLiteral("buttonBox"));
        buttonBox->setGeometry(QRect(30, 240, 341, 32));
        buttonBox->setOrientation(Qt::Horizontal);
        buttonBox->setStandardButtons(QDialogButtonBox::Cancel|QDialogButtonBox::Ok);
        tabWidget = new QTabWidget(OperatorWindow);
        tabWidget->setObjectName(QStringLiteral("tabWidget"));
        tabWidget->setGeometry(QRect(20, 10, 127, 80));
        tab = new QWidget();
        tab->setObjectName(QStringLiteral("tab"));
        tabWidget->addTab(tab, QString());
        tab_2 = new QWidget();
        tab_2->setObjectName(QStringLiteral("tab_2"));
        tabWidget->addTab(tab_2, QString());

        retranslateUi(OperatorWindow);
        QObject::connect(buttonBox, SIGNAL(accepted()), OperatorWindow, SLOT(accept()));
        QObject::connect(buttonBox, SIGNAL(rejected()), OperatorWindow, SLOT(reject()));

        QMetaObject::connectSlotsByName(OperatorWindow);
    } // setupUi

    void retranslateUi(QDialog *OperatorWindow)
    {
        OperatorWindow->setWindowTitle(QApplication::translate("OperatorWindow", "Dialog", 0));
        tabWidget->setTabText(tabWidget->indexOf(tab), QApplication::translate("OperatorWindow", "Tab 1", 0));
        tabWidget->setTabText(tabWidget->indexOf(tab_2), QApplication::translate("OperatorWindow", "Tab 2", 0));
    } // retranslateUi

};

namespace Ui {
    class OperatorWindow: public Ui_OperatorWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_OPERATORWINDOW_H
