/****************************************************************************
** Meta object code from reading C++ file 'GraphUnitTest.hpp'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.6.0)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "GraphUnitTest.hpp"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'GraphUnitTest.hpp' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.6.0. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
struct qt_meta_stringdata_GraphUnitTest_t {
    QByteArrayData data[10];
    char stringdata0[180];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_GraphUnitTest_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_GraphUnitTest_t qt_meta_stringdata_GraphUnitTest = {
    {
QT_MOC_LITERAL(0, 0, 13), // "GraphUnitTest"
QT_MOC_LITERAL(1, 14, 12), // "initTestCase"
QT_MOC_LITERAL(2, 27, 0), // ""
QT_MOC_LITERAL(3, 28, 4), // "init"
QT_MOC_LITERAL(4, 33, 36), // "Graph_CreateContext_CreationP..."
QT_MOC_LITERAL(5, 70, 27), // "Graph_Verification_Negative"
QT_MOC_LITERAL(6, 98, 29), // "Graph_DoVerification_Positive"
QT_MOC_LITERAL(7, 128, 27), // "Graph_Verification_Positive"
QT_MOC_LITERAL(8, 156, 7), // "cleanup"
QT_MOC_LITERAL(9, 164, 15) // "cleanupTestCase"

    },
    "GraphUnitTest\0initTestCase\0\0init\0"
    "Graph_CreateContext_CreationPositive\0"
    "Graph_Verification_Negative\0"
    "Graph_DoVerification_Positive\0"
    "Graph_Verification_Positive\0cleanup\0"
    "cleanupTestCase"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_GraphUnitTest[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       8,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    0,   54,    2, 0x08 /* Private */,
       3,    0,   55,    2, 0x08 /* Private */,
       4,    0,   56,    2, 0x08 /* Private */,
       5,    0,   57,    2, 0x08 /* Private */,
       6,    0,   58,    2, 0x08 /* Private */,
       7,    0,   59,    2, 0x08 /* Private */,
       8,    0,   60,    2, 0x08 /* Private */,
       9,    0,   61,    2, 0x08 /* Private */,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,

       0        // eod
};

void GraphUnitTest::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        GraphUnitTest *_t = static_cast<GraphUnitTest *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->initTestCase(); break;
        case 1: _t->init(); break;
        case 2: _t->Graph_CreateContext_CreationPositive(); break;
        case 3: _t->Graph_Verification_Negative(); break;
        case 4: _t->Graph_DoVerification_Positive(); break;
        case 5: _t->Graph_Verification_Positive(); break;
        case 6: _t->cleanup(); break;
        case 7: _t->cleanupTestCase(); break;
        default: ;
        }
    }
    Q_UNUSED(_a);
}

const QMetaObject GraphUnitTest::staticMetaObject = {
    { &TestSuite::staticMetaObject, qt_meta_stringdata_GraphUnitTest.data,
      qt_meta_data_GraphUnitTest,  qt_static_metacall, Q_NULLPTR, Q_NULLPTR}
};


const QMetaObject *GraphUnitTest::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *GraphUnitTest::qt_metacast(const char *_clname)
{
    if (!_clname) return Q_NULLPTR;
    if (!strcmp(_clname, qt_meta_stringdata_GraphUnitTest.stringdata0))
        return static_cast<void*>(const_cast< GraphUnitTest*>(this));
    return TestSuite::qt_metacast(_clname);
}

int GraphUnitTest::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = TestSuite::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 8)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 8;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 8)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 8;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
