/****************************************************************************
** Meta object code from reading C++ file 'VPL_Viewer.hpp'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.6.0)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "VPL_Viewer.h"
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
struct qt_meta_stringdata_RW__VPL__VPL_Viewer_t {
    QByteArrayData data[15];
    char stringdata0[174];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_RW__VPL__VPL_Viewer_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_RW__VPL__VPL_Viewer_t qt_meta_stringdata_RW__VPL__VPL_Viewer = {
    {
QT_MOC_LITERAL(0, 0, 19), // "RW::VPL::VPL_Viewer"
QT_MOC_LITERAL(1, 20, 12), // "setVideoData"
QT_MOC_LITERAL(2, 33, 0), // ""
QT_MOC_LITERAL(3, 34, 8), // "QBuffer*"
QT_MOC_LITERAL(4, 43, 7), // "qBuffer"
QT_MOC_LITERAL(5, 51, 4), // "play"
QT_MOC_LITERAL(6, 56, 17), // "mediaStateChanged"
QT_MOC_LITERAL(7, 74, 19), // "QMediaPlayer::State"
QT_MOC_LITERAL(8, 94, 5), // "state"
QT_MOC_LITERAL(9, 100, 15), // "positionChanged"
QT_MOC_LITERAL(10, 116, 8), // "position"
QT_MOC_LITERAL(11, 125, 15), // "durationChanged"
QT_MOC_LITERAL(12, 141, 8), // "duration"
QT_MOC_LITERAL(13, 150, 11), // "setPosition"
QT_MOC_LITERAL(14, 162, 11) // "handleError"

    },
    "RW::VPL::VPL_Viewer\0setVideoData\0\0"
    "QBuffer*\0qBuffer\0play\0mediaStateChanged\0"
    "QMediaPlayer::State\0state\0positionChanged\0"
    "position\0durationChanged\0duration\0"
    "setPosition\0handleError"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_RW__VPL__VPL_Viewer[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       7,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    1,   49,    2, 0x0a /* Public */,
       5,    0,   52,    2, 0x0a /* Public */,
       6,    1,   53,    2, 0x08 /* Private */,
       9,    1,   56,    2, 0x08 /* Private */,
      11,    1,   59,    2, 0x08 /* Private */,
      13,    1,   62,    2, 0x08 /* Private */,
      14,    0,   65,    2, 0x08 /* Private */,

 // slots: parameters
    QMetaType::Void, 0x80000000 | 3,    4,
    QMetaType::Void,
    QMetaType::Void, 0x80000000 | 7,    8,
    QMetaType::Void, QMetaType::LongLong,   10,
    QMetaType::Void, QMetaType::LongLong,   12,
    QMetaType::Void, QMetaType::Int,   10,
    QMetaType::Void,

       0        // eod
};

void RW::VPL::VPL_Viewer::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        VPL_Viewer *_t = static_cast<VPL_Viewer *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->setVideoData((*reinterpret_cast< QBuffer*(*)>(_a[1]))); break;
        case 1: _t->play(); break;
        case 2: _t->mediaStateChanged((*reinterpret_cast< QMediaPlayer::State(*)>(_a[1]))); break;
        case 3: _t->positionChanged((*reinterpret_cast< qint64(*)>(_a[1]))); break;
        case 4: _t->durationChanged((*reinterpret_cast< qint64(*)>(_a[1]))); break;
        case 5: _t->setPosition((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 6: _t->handleError(); break;
        default: ;
        }
    }
}

const QMetaObject RW::VPL::VPL_Viewer::staticMetaObject = {
    { &QWidget::staticMetaObject, qt_meta_stringdata_RW__VPL__VPL_Viewer.data,
      qt_meta_data_RW__VPL__VPL_Viewer,  qt_static_metacall, Q_NULLPTR, Q_NULLPTR}
};


const QMetaObject *RW::VPL::VPL_Viewer::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *RW::VPL::VPL_Viewer::qt_metacast(const char *_clname)
{
    if (!_clname) return Q_NULLPTR;
    if (!strcmp(_clname, qt_meta_stringdata_RW__VPL__VPL_Viewer.stringdata0))
        return static_cast<void*>(const_cast< VPL_Viewer*>(this));
    return QWidget::qt_metacast(_clname);
}

int RW::VPL::VPL_Viewer::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 7)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 7;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 7)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 7;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
