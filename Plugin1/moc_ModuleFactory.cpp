/****************************************************************************
** Meta object code from reading C++ file 'ModuleFactory.hpp'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.6.0)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "ModuleFactory.hpp"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#include <QtCore/qplugin.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'ModuleFactory.hpp' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.6.0. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
struct qt_meta_stringdata_RW__TEST__ModuleFactory_t {
    QByteArrayData data[1];
    char stringdata0[24];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_RW__TEST__ModuleFactory_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_RW__TEST__ModuleFactory_t qt_meta_stringdata_RW__TEST__ModuleFactory = {
    {
QT_MOC_LITERAL(0, 0, 23) // "RW::TEST::ModuleFactory"

    },
    "RW::TEST::ModuleFactory"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_RW__TEST__ModuleFactory[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       0,    0, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

       0        // eod
};

void RW::TEST::ModuleFactory::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    Q_UNUSED(_o);
    Q_UNUSED(_id);
    Q_UNUSED(_c);
    Q_UNUSED(_a);
}

const QMetaObject RW::TEST::ModuleFactory::staticMetaObject = {
    { &CORE::AbstractModuleFactory::staticMetaObject, qt_meta_stringdata_RW__TEST__ModuleFactory.data,
      qt_meta_data_RW__TEST__ModuleFactory,  qt_static_metacall, Q_NULLPTR, Q_NULLPTR}
};


const QMetaObject *RW::TEST::ModuleFactory::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *RW::TEST::ModuleFactory::qt_metacast(const char *_clname)
{
    if (!_clname) return Q_NULLPTR;
    if (!strcmp(_clname, qt_meta_stringdata_RW__TEST__ModuleFactory.stringdata0))
        return static_cast<void*>(const_cast< ModuleFactory*>(this));
    if (!strcmp(_clname, "AbstractModuleFactory"))
        return static_cast< RW::CORE::AbstractModuleFactory*>(const_cast< ModuleFactory*>(this));
    return CORE::AbstractModuleFactory::qt_metacast(_clname);
}

int RW::TEST::ModuleFactory::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = CORE::AbstractModuleFactory::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    return _id;
}

QT_PLUGIN_METADATA_SECTION const uint qt_section_alignment_dummy = 42;

#ifdef QT_NO_DEBUG

QT_PLUGIN_METADATA_SECTION
static const unsigned char qt_pluginMetaData[] = {
    'Q', 'T', 'M', 'E', 'T', 'A', 'D', 'A', 'T', 'A', ' ', ' ',
    'q',  'b',  'j',  's',  0x01, 0x00, 0x00, 0x00,
    0xb8, 0x00, 0x00, 0x00, 0x0b, 0x00, 0x00, 0x00,
    0xa4, 0x00, 0x00, 0x00, 0x1b, 0x03, 0x00, 0x00,
    0x03, 0x00, 'I',  'I',  'D',  0x00, 0x00, 0x00,
    0x15, 0x00, 'A',  'b',  's',  't',  'r',  'a', 
    'c',  't',  'M',  'o',  'd',  'u',  'l',  'e', 
    'F',  'a',  'c',  't',  'o',  'r',  'y',  0x00,
    0x1b, 0x08, 0x00, 0x00, 0x09, 0x00, 'c',  'l', 
    'a',  's',  's',  'N',  'a',  'm',  'e',  0x00,
    0x0d, 0x00, 'M',  'o',  'd',  'u',  'l',  'e', 
    'F',  'a',  'c',  't',  'o',  'r',  'y',  0x00,
    0x1a, 0xc0, 0xa0, 0x00, 0x07, 0x00, 'v',  'e', 
    'r',  's',  'i',  'o',  'n',  0x00, 0x00, 0x00,
    0x11, 0x00, 0x00, 0x00, 0x05, 0x00, 'd',  'e', 
    'b',  'u',  'g',  0x00, 0x95, 0x0f, 0x00, 0x00,
    0x08, 0x00, 'M',  'e',  't',  'a',  'D',  'a', 
    't',  'a',  0x00, 0x00, '(',  0x00, 0x00, 0x00,
    0x03, 0x00, 0x00, 0x00, '$',  0x00, 0x00, 0x00,
    0x14, 0x03, 0x00, 0x00, 0x04, 0x00, 'K',  'e', 
    'y',  's',  0x00, 0x00, 0x0c, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x0c, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00,
    'l',  0x00, 0x00, 0x00, '0',  0x00, 0x00, 0x00,
    '`',  0x00, 0x00, 0x00, 'P',  0x00, 0x00, 0x00
};

#else // QT_NO_DEBUG

QT_PLUGIN_METADATA_SECTION
static const unsigned char qt_pluginMetaData[] = {
    'Q', 'T', 'M', 'E', 'T', 'A', 'D', 'A', 'T', 'A', ' ', ' ',
    'q',  'b',  'j',  's',  0x01, 0x00, 0x00, 0x00,
    0xb8, 0x00, 0x00, 0x00, 0x0b, 0x00, 0x00, 0x00,
    0xa4, 0x00, 0x00, 0x00, 0x1b, 0x03, 0x00, 0x00,
    0x03, 0x00, 'I',  'I',  'D',  0x00, 0x00, 0x00,
    0x15, 0x00, 'A',  'b',  's',  't',  'r',  'a', 
    'c',  't',  'M',  'o',  'd',  'u',  'l',  'e', 
    'F',  'a',  'c',  't',  'o',  'r',  'y',  0x00,
    0x15, 0x08, 0x00, 0x00, 0x08, 0x00, 'M',  'e', 
    't',  'a',  'D',  'a',  't',  'a',  0x00, 0x00,
    '(',  0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
    '$',  0x00, 0x00, 0x00, 0x14, 0x03, 0x00, 0x00,
    0x04, 0x00, 'K',  'e',  'y',  's',  0x00, 0x00,
    0x0c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00,
    0x1b, 0x0f, 0x00, 0x00, 0x09, 0x00, 'c',  'l', 
    'a',  's',  's',  'N',  'a',  'm',  'e',  0x00,
    0x0d, 0x00, 'M',  'o',  'd',  'u',  'l',  'e', 
    'F',  'a',  'c',  't',  'o',  'r',  'y',  0x00,
    '1',  0x00, 0x00, 0x00, 0x05, 0x00, 'd',  'e', 
    'b',  'u',  'g',  0x00, 0x1a, 0xc0, 0xa0, 0x00,
    0x07, 0x00, 'v',  'e',  'r',  's',  'i',  'o', 
    'n',  0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00,
    '0',  0x00, 0x00, 0x00, 'h',  0x00, 0x00, 0x00,
    0x88, 0x00, 0x00, 0x00, 0x94, 0x00, 0x00, 0x00
};
#endif // QT_NO_DEBUG

using namespace RW;
using namespace RW::TEST;
QT_MOC_EXPORT_PLUGIN(RW::TEST::ModuleFactory, ModuleFactory)

QT_END_MOC_NAMESPACE
