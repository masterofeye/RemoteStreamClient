/****************************************************************************
** Meta object code from reading C++ file 'ContextUnitTest.hpp'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.6.0)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "ContextUnitTest.hpp"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'ContextUnitTest.hpp' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.6.0. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
struct qt_meta_stringdata_ContextUnitTest_t {
    QByteArrayData data[9];
    char stringdata0[151];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_ContextUnitTest_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_ContextUnitTest_t qt_meta_stringdata_ContextUnitTest = {
    {
QT_MOC_LITERAL(0, 0, 15), // "ContextUnitTest"
QT_MOC_LITERAL(1, 16, 12), // "initTestCase"
QT_MOC_LITERAL(2, 29, 0), // ""
QT_MOC_LITERAL(3, 30, 4), // "init"
QT_MOC_LITERAL(4, 35, 38), // "Context_CreateContext_Creatio..."
QT_MOC_LITERAL(5, 74, 25), // "Context_GetVendor_Positiv"
QT_MOC_LITERAL(6, 100, 26), // "Context_GetVersion_Positiv"
QT_MOC_LITERAL(7, 127, 7), // "cleanup"
QT_MOC_LITERAL(8, 135, 15) // "cleanupTestCase"

    },
    "ContextUnitTest\0initTestCase\0\0init\0"
    "Context_CreateContext_CreationPositive\0"
    "Context_GetVendor_Positiv\0"
    "Context_GetVersion_Positiv\0cleanup\0"
    "cleanupTestCase"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_ContextUnitTest[] = {

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
       4,    0,   51,    2, 0x08 /* Private */,
       5,    0,   52,    2, 0x08 /* Private */,
       6,    0,   53,    2, 0x08 /* Private */,
       7,    0,   54,    2, 0x08 /* Private */,
       8,    0,   55,    2, 0x08 /* Private */,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,

       0        // eod
};

void ContextUnitTest::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        ContextUnitTest *_t = static_cast<ContextUnitTest *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->initTestCase(); break;
        case 1: _t->init(); break;
        case 2: _t->Context_CreateContext_CreationPositive(); break;
        case 3: _t->Context_GetVendor_Positiv(); break;
        case 4: _t->Context_GetVersion_Positiv(); break;
        case 5: _t->cleanup(); break;
        case 6: _t->cleanupTestCase(); break;
        default: ;
        }
    }
    Q_UNUSED(_a);
}

const QMetaObject ContextUnitTest::staticMetaObject = {
    { &TestSuite::staticMetaObject, qt_meta_stringdata_ContextUnitTest.data,
      qt_meta_data_ContextUnitTest,  qt_static_metacall, Q_NULLPTR, Q_NULLPTR}
};


const QMetaObject *ContextUnitTest::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *ContextUnitTest::qt_metacast(const char *_clname)
{
    if (!_clname) return Q_NULLPTR;
    if (!strcmp(_clname, qt_meta_stringdata_ContextUnitTest.stringdata0))
        return static_cast<void*>(const_cast< ContextUnitTest*>(this));
    return TestSuite::qt_metacast(_clname);
}

int ContextUnitTest::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = TestSuite::qt_metacall(_c, _id, _a);
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
