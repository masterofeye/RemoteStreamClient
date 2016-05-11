/****************************************************************************
** Meta object code from reading C++ file 'AbstractModule.hpp'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.6.0)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "AbstractModule.hpp"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'AbstractModule.hpp' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.6.0. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
struct qt_meta_stringdata_RW__CORE__AbstractModule_t {
    QByteArrayData data[15];
    char stringdata0[236];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_RW__CORE__AbstractModule_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_RW__CORE__AbstractModule_t qt_meta_stringdata_RW__CORE__AbstractModule = {
    {
QT_MOC_LITERAL(0, 0, 24), // "RW::CORE::AbstractModule"
QT_MOC_LITERAL(1, 25, 8), // "Finished"
QT_MOC_LITERAL(2, 34, 0), // ""
QT_MOC_LITERAL(3, 35, 12), // "ModulVersion"
QT_MOC_LITERAL(4, 48, 22), // "CORE::tstModuleVersion"
QT_MOC_LITERAL(5, 71, 12), // "SubModulType"
QT_MOC_LITERAL(6, 84, 18), // "CORE::tenSubModule"
QT_MOC_LITERAL(7, 103, 10), // "Initialise"
QT_MOC_LITERAL(8, 114, 9), // "tenStatus"
QT_MOC_LITERAL(9, 124, 27), // "tstInitialiseControlStruct*"
QT_MOC_LITERAL(10, 152, 13), // "ControlStruct"
QT_MOC_LITERAL(11, 166, 8), // "DoRender"
QT_MOC_LITERAL(12, 175, 17), // "tstControlStruct*"
QT_MOC_LITERAL(13, 193, 12), // "Deinitialise"
QT_MOC_LITERAL(14, 206, 29) // "tstDeinitialiseControlStruct*"

    },
    "RW::CORE::AbstractModule\0Finished\0\0"
    "ModulVersion\0CORE::tstModuleVersion\0"
    "SubModulType\0CORE::tenSubModule\0"
    "Initialise\0tenStatus\0tstInitialiseControlStruct*\0"
    "ControlStruct\0DoRender\0tstControlStruct*\0"
    "Deinitialise\0tstDeinitialiseControlStruct*"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_RW__CORE__AbstractModule[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       6,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    0,   44,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
       3,    0,   45,    2, 0x0a /* Public */,
       5,    0,   46,    2, 0x0a /* Public */,
       7,    1,   47,    2, 0x0a /* Public */,
      11,    1,   50,    2, 0x0a /* Public */,
      13,    1,   53,    2, 0x0a /* Public */,

 // signals: parameters
    QMetaType::Void,

 // slots: parameters
    0x80000000 | 4,
    0x80000000 | 6,
    0x80000000 | 8, 0x80000000 | 9,   10,
    0x80000000 | 8, 0x80000000 | 12,   10,
    0x80000000 | 8, 0x80000000 | 14,   10,

       0        // eod
};

void RW::CORE::AbstractModule::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        AbstractModule *_t = static_cast<AbstractModule *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->Finished(); break;
        case 1: { CORE::tstModuleVersion _r = _t->ModulVersion();
            if (_a[0]) *reinterpret_cast< CORE::tstModuleVersion*>(_a[0]) = _r; }  break;
        case 2: { CORE::tenSubModule _r = _t->SubModulType();
            if (_a[0]) *reinterpret_cast< CORE::tenSubModule*>(_a[0]) = _r; }  break;
        case 3: { tenStatus _r = _t->Initialise((*reinterpret_cast< tstInitialiseControlStruct*(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< tenStatus*>(_a[0]) = _r; }  break;
        case 4: { tenStatus _r = _t->DoRender((*reinterpret_cast< tstControlStruct*(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< tenStatus*>(_a[0]) = _r; }  break;
        case 5: { tenStatus _r = _t->Deinitialise((*reinterpret_cast< tstDeinitialiseControlStruct*(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< tenStatus*>(_a[0]) = _r; }  break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        void **func = reinterpret_cast<void **>(_a[1]);
        {
            typedef void (AbstractModule::*_t)();
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&AbstractModule::Finished)) {
                *result = 0;
                return;
            }
        }
    }
}

const QMetaObject RW::CORE::AbstractModule::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_RW__CORE__AbstractModule.data,
      qt_meta_data_RW__CORE__AbstractModule,  qt_static_metacall, Q_NULLPTR, Q_NULLPTR}
};


const QMetaObject *RW::CORE::AbstractModule::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *RW::CORE::AbstractModule::qt_metacast(const char *_clname)
{
    if (!_clname) return Q_NULLPTR;
    if (!strcmp(_clname, qt_meta_stringdata_RW__CORE__AbstractModule.stringdata0))
        return static_cast<void*>(const_cast< AbstractModule*>(this));
    return QObject::qt_metacast(_clname);
}

int RW::CORE::AbstractModule::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 6)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 6;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 6)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 6;
    }
    return _id;
}

// SIGNAL 0
void RW::CORE::AbstractModule::Finished()
{
    QMetaObject::activate(this, &staticMetaObject, 0, Q_NULLPTR);
}
QT_END_MOC_NAMESPACE
