QT -= qt core gui

TARGET = KitsunemimiOpencl
TEMPLATE = lib
CONFIG += c++14
VERSION = 0.3.0

LIBS += -L../../libKitsunemimiCommon/src -lKitsunemimiCommon
LIBS += -L../../libKitsunemimiCommon/src/debug -lKitsunemimiCommon
LIBS += -L../../libKitsunemimiCommon/src/release -lKitsunemimiCommon
INCLUDEPATH += ../../libKitsunemimiCommon/include

LIBS += -L../../libKitsunemimiPersistence/src -lKitsunemimiPersistence
LIBS += -L../../libKitsunemimiPersistence/src/debug -lKitsunemimiPersistence
LIBS += -L../../libKitsunemimiPersistence/src/release -lKitsunemimiPersistence
INCLUDEPATH += ../../libKitsunemimiPersistence/include

LIBS +=  -lOpenCL -lboost_filesystem -lboost_system


INCLUDEPATH += $$PWD \
               $$PWD/../include

HEADERS += \
    ../include/libKitsunemimiOpencl/gpu_interface.h \
    ../include/libKitsunemimiOpencl/gpu_handler.h \
    ../include/libKitsunemimiOpencl/gpu_data.h

SOURCES += \
    gpu_interface.cpp \
    gpu_handler.cpp \
    gpu_data.cpp

unix {
    INCLUDEPATH += /usr/lib/gcc/x86_64-linux-gnu/9/include
}
