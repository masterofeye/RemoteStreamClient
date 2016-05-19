/****************************************************************************
** Meta object code from reading C++ file 'VideoPlayer.hpp'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.6.0)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "VideoPlayer.hpp"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'VideoPlayer.hpp' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.6.0. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
struct qt_meta_stringdata_RW__VPL__VideoPlayer_t {
    QByteArrayData data[13];
    char stringdata0[154];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_RW__VPL__VideoPlayer_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_RW__VPL__VideoPlayer_t qt_meta_stringdata_RW__VPL__VideoPlayer = {
    {
QT_MOC_LITERAL(0, 0, 20), // "RW::VPL::VideoPlayer"
QT_MOC_LITERAL(1, 21, 8), // "openFile"
QT_MOC_LITERAL(2, 30, 0), // ""
QT_MOC_LITERAL(3, 31, 4), // "play"
QT_MOC_LITERAL(4, 36, 17), // "mediaStateChanged"
QT_MOC_LITERAL(5, 54, 19), // "QMediaPlayer::State"
QT_MOC_LITERAL(6, 74, 5), // "state"
QT_MOC_LITERAL(7, 80, 15), // "positionChanged"
QT_MOC_LITERAL(8, 96, 8), // "position"
QT_MOC_LITERAL(9, 105, 15), // "durationChanged"
QT_MOC_LITERAL(10, 121, 8), // "duration"
QT_MOC_LITERAL(11, 130, 11), // "setPosition"
QT_MOC_LITERAL(12, 142, 11) // "handleError"

    },
    "RW::VPL::VideoPlayer\0openFile\0\0play\0"
    "mediaStateChanged\0QMediaPlayer::State\0"
    "state\0positionChanged\0position\0"
    "durationChanged\0duration\0setPosition\0"
    "handleError"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_RW__VPL__VideoPlayer[] = {

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
       1,    0,   49,    2, 0x08 /* Private */,
       3,    0,   50,    2, 0x08 /* Private */,
       4,    1,   51,    2, 0x08 /* Private */,
       7,    1,   54,    2, 0x08 /* Private */,
       9,    1,   57,    2, 0x08 /* Private */,
      11,    1,   60,    2, 0x08 /* Private */,
      12,    0,   63,    2, 0x08 /* Private */,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, 0x80000000 | 5,    6,
    QMetaType::Void, QMetaType::LongLong,    8,
    QMetaType::Void, QMetaType::LongLong,   10,
    QMetaType::Void, QMetaType::Int,    8,
    QMetaType::Void,

       0        // eod
};

void RW::VPL::VideoPlayer::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        VideoPlayer *_t = static_cast<VideoPlayer *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->openFile(); break;
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

const QMetaObject RW::VPL::VideoPlayer::staticMetaObject = {
    { &RW::CORE::AbstractModule::staticMetaObject, qt_meta_stringdata_RW__VPL__VideoPlayer.data,
      qt_meta_data_RW__VPL__VideoPlayer,  qt_static_metacall, Q_NULLPTR, Q_NULLPTR}
};


const QMetaObject *RW::VPL::VideoPlayer::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *RW::VPL::VideoPlayer::qt_metacast(const char *_clname)
{
    if (!_clname) return Q_NULLPTR;
    if (!strcmp(_clname, qt_meta_stringdata_RW__VPL__VideoPlayer.stringdata0))
        return static_cast<void*>(const_cast< VideoPlayer*>(this));
    return RW::CORE::AbstractModule::qt_metacast(_clname);
}

int RW::VPL::VideoPlayer::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = RW::CORE::AbstractModule::qt_metacall(_c, _id, _a);
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
