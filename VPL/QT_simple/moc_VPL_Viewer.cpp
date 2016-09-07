/****************************************************************************
** Meta object code from reading C++ file 'VPL_Viewer.hpp'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.6.0)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "VPL_Viewer.hpp"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'VPL_Viewer.hpp' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.6.0. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
struct qt_meta_stringdata_RW__VPL__QT_SIMPLE__VPL_Viewer_t {
    QByteArrayData data[7];
    char stringdata0[98];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_RW__VPL__QT_SIMPLE__VPL_Viewer_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_RW__VPL__QT_SIMPLE__VPL_Viewer_t qt_meta_stringdata_RW__VPL__QT_SIMPLE__VPL_Viewer = {
    {
QT_MOC_LITERAL(0, 0, 30), // "RW::VPL::QT_SIMPLE::VPL_Viewer"
QT_MOC_LITERAL(1, 31, 12), // "setVideoData"
QT_MOC_LITERAL(2, 44, 0), // ""
QT_MOC_LITERAL(3, 45, 6), // "buffer"
QT_MOC_LITERAL(4, 52, 15), // "connectToViewer"
QT_MOC_LITERAL(5, 68, 19), // "VPL_FrameProcessor*"
QT_MOC_LITERAL(6, 88, 9) // "frameProc"

    },
    "RW::VPL::QT_SIMPLE::VPL_Viewer\0"
    "setVideoData\0\0buffer\0connectToViewer\0"
    "VPL_FrameProcessor*\0frameProc"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_RW__VPL__QT_SIMPLE__VPL_Viewer[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       2,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    1,   24,    2, 0x0a /* Public */,
       4,    1,   27,    2, 0x0a /* Public */,

 // slots: parameters
    QMetaType::Void, QMetaType::VoidStar,    3,
    QMetaType::Void, 0x80000000 | 5,    6,

       0        // eod
};

void RW::VPL::QT_SIMPLE::VPL_Viewer::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        VPL_Viewer *_t = static_cast<VPL_Viewer *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->setVideoData((*reinterpret_cast< void*(*)>(_a[1]))); break;
        case 1: _t->connectToViewer((*reinterpret_cast< VPL_FrameProcessor*(*)>(_a[1]))); break;
        default: ;
        }
    }
}

const QMetaObject RW::VPL::QT_SIMPLE::VPL_Viewer::staticMetaObject = {
    { &QWidget::staticMetaObject, qt_meta_stringdata_RW__VPL__QT_SIMPLE__VPL_Viewer.data,
      qt_meta_data_RW__VPL__QT_SIMPLE__VPL_Viewer,  qt_static_metacall, Q_NULLPTR, Q_NULLPTR}
};


const QMetaObject *RW::VPL::QT_SIMPLE::VPL_Viewer::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *RW::VPL::QT_SIMPLE::VPL_Viewer::qt_metacast(const char *_clname)
{
    if (!_clname) return Q_NULLPTR;
    if (!strcmp(_clname, qt_meta_stringdata_RW__VPL__QT_SIMPLE__VPL_Viewer.stringdata0))
        return static_cast<void*>(const_cast< VPL_Viewer*>(this));
    return QWidget::qt_metacast(_clname);
}

int RW::VPL::QT_SIMPLE::VPL_Viewer::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 2)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 2;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 2)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 2;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
