/****************************************************************************
** Meta object code from reading C++ file 'Pipeline.hpp'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.6.0)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "Pipeline.hpp"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'Pipeline.hpp' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.6.0. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
struct qt_meta_stringdata_CPipeline_t {
    QByteArrayData data[3];
    char stringdata0[23];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_CPipeline_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_CPipeline_t qt_meta_stringdata_CPipeline = {
    {
QT_MOC_LITERAL(0, 0, 9), // "CPipeline"
QT_MOC_LITERAL(1, 10, 11), // "RunPipeline"
QT_MOC_LITERAL(2, 22, 0) // ""

    },
    "CPipeline\0RunPipeline\0"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_CPipeline[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       1,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    0,   19,    2, 0x0a /* Public */,

 // slots: parameters
    QMetaType::Int,

       0        // eod
};

void CPipeline::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        CPipeline *_t = static_cast<CPipeline *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: { int _r = _t->RunPipeline();
            if (_a[0]) *reinterpret_cast< int*>(_a[0]) = _r; }  break;
        default: ;
        }
    }
}

const QMetaObject CPipeline::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_CPipeline.data,
      qt_meta_data_CPipeline,  qt_static_metacall, Q_NULLPTR, Q_NULLPTR}
};


const QMetaObject *CPipeline::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *CPipeline::qt_metacast(const char *_clname)
{
    if (!_clname) return Q_NULLPTR;
    if (!strcmp(_clname, qt_meta_stringdata_CPipeline.stringdata0))
        return static_cast<void*>(const_cast< CPipeline*>(this));
    return QObject::qt_metacast(_clname);
}

int CPipeline::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 1)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 1;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 1)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 1;
    }
    return _id;
}
struct qt_meta_stringdata_CPipethread_t {
    QByteArrayData data[3];
    char stringdata0[21];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_CPipethread_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_CPipethread_t qt_meta_stringdata_CPipethread = {
    {
QT_MOC_LITERAL(0, 0, 11), // "CPipethread"
QT_MOC_LITERAL(1, 12, 7), // "started"
QT_MOC_LITERAL(2, 20, 0) // ""

    },
    "CPipethread\0started\0"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_CPipethread[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       1,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    0,   19,    2, 0x06 /* Public */,

 // signals: parameters
    QMetaType::Int,

       0        // eod
};

void CPipethread::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        CPipethread *_t = static_cast<CPipethread *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: { int _r = _t->started();
            if (_a[0]) *reinterpret_cast< int*>(_a[0]) = _r; }  break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        void **func = reinterpret_cast<void **>(_a[1]);
        {
            typedef int (CPipethread::*_t)();
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&CPipethread::started)) {
                *result = 0;
                return;
            }
        }
    }
}

const QMetaObject CPipethread::staticMetaObject = {
    { &QThread::staticMetaObject, qt_meta_stringdata_CPipethread.data,
      qt_meta_data_CPipethread,  qt_static_metacall, Q_NULLPTR, Q_NULLPTR}
};


const QMetaObject *CPipethread::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *CPipethread::qt_metacast(const char *_clname)
{
    if (!_clname) return Q_NULLPTR;
    if (!strcmp(_clname, qt_meta_stringdata_CPipethread.stringdata0))
        return static_cast<void*>(const_cast< CPipethread*>(this));
    return QThread::qt_metacast(_clname);
}

int CPipethread::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QThread::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 1)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 1;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 1)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 1;
    }
    return _id;
}

// SIGNAL 0
int CPipethread::started()
{
    int _t0 = int();
    void *_a[] = { const_cast<void*>(reinterpret_cast<const void*>(&_t0)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
    return _t0;
}
QT_END_MOC_NAMESPACE
