include(../../defaults.pri)

QT -= qt core gui

CONFIG   -= app_bundle
CONFIG += c++14 console

LIBS +=  -lOpenCL

INCLUDEPATH += $$PWD

LIBS += -L../../src -lKitsunemimiOpencl

SOURCES += \
    main.cpp
