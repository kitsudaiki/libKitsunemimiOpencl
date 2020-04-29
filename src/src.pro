QT -= qt core gui

TARGET = KitsunemimiOpencl
TEMPLATE = lib
CONFIG += staticlib
CONFIG += c++14

LIBS +=  -lOpenCL

INCLUDEPATH += $$PWD \
               $$PWD/../include


HEADERS += \
    ../include/libKitsunemimiOpencl/opencl.h

SOURCES += \
    opencl.cpp
