#-------------------------------------------------
#
# Project created by QtCreator 2016-10-13T09:18:25
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = RSAPP
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
    appservice.cpp

HEADERS  += mainwindow.h \
    appservice.h

FORMS    += mainwindow.ui

TEMPLATE=vcapp
