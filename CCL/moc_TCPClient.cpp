/****************************************************************************
** Meta object code from reading C++ file 'TCPClient.hpp'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.6.0)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "TCPClient.hpp"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'TCPClient.hpp' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.6.0. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
struct qt_meta_stringdata_RW__CCL__TCPClient_t {
    QByteArrayData data[26];
    char stringdata0[321];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_RW__CCL__TCPClient_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_RW__CCL__TCPClient_t qt_meta_stringdata_RW__CCL__TCPClient = {
    {
QT_MOC_LITERAL(0, 0, 18), // "RW::CCL::TCPClient"
QT_MOC_LITERAL(1, 19, 20), // "SetConfigForPipeline"
QT_MOC_LITERAL(2, 40, 0), // ""
QT_MOC_LITERAL(3, 41, 13), // "sSimpleConfig"
QT_MOC_LITERAL(4, 55, 15), // "stConfiguration"
QT_MOC_LITERAL(5, 71, 14), // "SetUDPAdresses"
QT_MOC_LITERAL(6, 86, 16), // "std::list<char*>"
QT_MOC_LITERAL(7, 103, 12), // "lstAddresses"
QT_MOC_LITERAL(8, 116, 11), // "RunPipeline"
QT_MOC_LITERAL(9, 128, 12), // "StopPipeline"
QT_MOC_LITERAL(10, 141, 7), // "started"
QT_MOC_LITERAL(11, 149, 17), // "iConnectToSession"
QT_MOC_LITERAL(12, 167, 6), // "char[]"
QT_MOC_LITERAL(13, 174, 2), // "IP"
QT_MOC_LITERAL(14, 177, 11), // "bIsOperator"
QT_MOC_LITERAL(15, 189, 11), // "iDisconnect"
QT_MOC_LITERAL(16, 201, 13), // "iRemoveClient"
QT_MOC_LITERAL(17, 215, 8), // "ClientID"
QT_MOC_LITERAL(18, 224, 14), // "oGetClientList"
QT_MOC_LITERAL(19, 239, 22), // "std::list<ClientInfo*>"
QT_MOC_LITERAL(20, 262, 10), // "vSetConfig"
QT_MOC_LITERAL(21, 273, 8), // "vApprove"
QT_MOC_LITERAL(22, 282, 11), // "ipToApprove"
QT_MOC_LITERAL(23, 294, 5), // "iStop"
QT_MOC_LITERAL(24, 300, 6), // "vStart"
QT_MOC_LITERAL(25, 307, 13) // "vStopReceived"

    },
    "RW::CCL::TCPClient\0SetConfigForPipeline\0"
    "\0sSimpleConfig\0stConfiguration\0"
    "SetUDPAdresses\0std::list<char*>\0"
    "lstAddresses\0RunPipeline\0StopPipeline\0"
    "started\0iConnectToSession\0char[]\0IP\0"
    "bIsOperator\0iDisconnect\0iRemoveClient\0"
    "ClientID\0oGetClientList\0std::list<ClientInfo*>\0"
    "vSetConfig\0vApprove\0ipToApprove\0iStop\0"
    "vStart\0vStopReceived"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_RW__CCL__TCPClient[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
      15,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       5,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    1,   89,    2, 0x06 /* Public */,
       5,    1,   92,    2, 0x06 /* Public */,
       8,    0,   95,    2, 0x06 /* Public */,
       9,    0,   96,    2, 0x06 /* Public */,
      10,    0,   97,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
      11,    2,   98,    2, 0x0a /* Public */,
      11,    1,  103,    2, 0x2a /* Public | MethodCloned */,
      15,    0,  106,    2, 0x0a /* Public */,
      16,    1,  107,    2, 0x0a /* Public */,
      18,    0,  110,    2, 0x0a /* Public */,
      20,    1,  111,    2, 0x0a /* Public */,
      21,    1,  114,    2, 0x0a /* Public */,
      23,    0,  117,    2, 0x0a /* Public */,
      24,    0,  118,    2, 0x0a /* Public */,
      25,    0,  119,    2, 0x0a /* Public */,

 // signals: parameters
    QMetaType::Void, 0x80000000 | 3,    4,
    QMetaType::Void, 0x80000000 | 6,    7,
    QMetaType::Int,
    QMetaType::Int,
    QMetaType::Void,

 // slots: parameters
    QMetaType::Int, 0x80000000 | 12, QMetaType::Bool,   13,   14,
    QMetaType::Int, 0x80000000 | 12,   13,
    QMetaType::Int,
    QMetaType::Int, 0x80000000 | 12,   17,
    0x80000000 | 19,
    QMetaType::Void, 0x80000000 | 3,    4,
    QMetaType::Void, 0x80000000 | 12,   22,
    QMetaType::Int,
    QMetaType::Void,
    QMetaType::Void,

       0        // eod
};

void RW::CCL::TCPClient::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        TCPClient *_t = static_cast<TCPClient *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->SetConfigForPipeline((*reinterpret_cast< sSimpleConfig(*)>(_a[1]))); break;
        case 1: _t->SetUDPAdresses((*reinterpret_cast< std::list<char*>(*)>(_a[1]))); break;
        case 2: { int _r = _t->RunPipeline();
            if (_a[0]) *reinterpret_cast< int*>(_a[0]) = _r; }  break;
        case 3: { int _r = _t->StopPipeline();
            if (_a[0]) *reinterpret_cast< int*>(_a[0]) = _r; }  break;
        case 4: _t->started(); break;
        case 5: { int _r = _t->iConnectToSession((*reinterpret_cast< char(*)[]>(_a[1])),(*reinterpret_cast< bool(*)>(_a[2])));
            if (_a[0]) *reinterpret_cast< int*>(_a[0]) = _r; }  break;
        case 6: { int _r = _t->iConnectToSession((*reinterpret_cast< char(*)[]>(_a[1])));
            if (_a[0]) *reinterpret_cast< int*>(_a[0]) = _r; }  break;
        case 7: { int _r = _t->iDisconnect();
            if (_a[0]) *reinterpret_cast< int*>(_a[0]) = _r; }  break;
        case 8: { int _r = _t->iRemoveClient((*reinterpret_cast< char(*)[]>(_a[1])));
            if (_a[0]) *reinterpret_cast< int*>(_a[0]) = _r; }  break;
        case 9: { std::list<ClientInfo*> _r = _t->oGetClientList();
            if (_a[0]) *reinterpret_cast< std::list<ClientInfo*>*>(_a[0]) = _r; }  break;
        case 10: _t->vSetConfig((*reinterpret_cast< sSimpleConfig(*)>(_a[1]))); break;
        case 11: _t->vApprove((*reinterpret_cast< char(*)[]>(_a[1]))); break;
        case 12: { int _r = _t->iStop();
            if (_a[0]) *reinterpret_cast< int*>(_a[0]) = _r; }  break;
        case 13: _t->vStart(); break;
        case 14: _t->vStopReceived(); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        void **func = reinterpret_cast<void **>(_a[1]);
        {
            typedef void (TCPClient::*_t)(sSimpleConfig );
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&TCPClient::SetConfigForPipeline)) {
                *result = 0;
                return;
            }
        }
        {
            typedef void (TCPClient::*_t)(std::list<char*> );
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&TCPClient::SetUDPAdresses)) {
                *result = 1;
                return;
            }
        }
        {
            typedef int (TCPClient::*_t)();
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&TCPClient::RunPipeline)) {
                *result = 2;
                return;
            }
        }
        {
            typedef int (TCPClient::*_t)();
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&TCPClient::StopPipeline)) {
                *result = 3;
                return;
            }
        }
        {
            typedef void (TCPClient::*_t)();
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&TCPClient::started)) {
                *result = 4;
                return;
            }
        }
    }
}

const QMetaObject RW::CCL::TCPClient::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_RW__CCL__TCPClient.data,
      qt_meta_data_RW__CCL__TCPClient,  qt_static_metacall, Q_NULLPTR, Q_NULLPTR}
};


const QMetaObject *RW::CCL::TCPClient::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *RW::CCL::TCPClient::qt_metacast(const char *_clname)
{
    if (!_clname) return Q_NULLPTR;
    if (!strcmp(_clname, qt_meta_stringdata_RW__CCL__TCPClient.stringdata0))
        return static_cast<void*>(const_cast< TCPClient*>(this));
    return QObject::qt_metacast(_clname);
}

int RW::CCL::TCPClient::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 15)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 15;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 15)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 15;
    }
    return _id;
}

// SIGNAL 0
void RW::CCL::TCPClient::SetConfigForPipeline(sSimpleConfig _t1)
{
    void *_a[] = { Q_NULLPTR, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void RW::CCL::TCPClient::SetUDPAdresses(std::list<char*> _t1)
{
    void *_a[] = { Q_NULLPTR, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}

// SIGNAL 2
int RW::CCL::TCPClient::RunPipeline()
{
    int _t0 = int();
    void *_a[] = { const_cast<void*>(reinterpret_cast<const void*>(&_t0)) };
    QMetaObject::activate(this, &staticMetaObject, 2, _a);
    return _t0;
}

// SIGNAL 3
int RW::CCL::TCPClient::StopPipeline()
{
    int _t0 = int();
    void *_a[] = { const_cast<void*>(reinterpret_cast<const void*>(&_t0)) };
    QMetaObject::activate(this, &staticMetaObject, 3, _a);
    return _t0;
}

// SIGNAL 4
void RW::CCL::TCPClient::started()
{
    QMetaObject::activate(this, &staticMetaObject, 4, Q_NULLPTR);
}
struct qt_meta_stringdata_RW__CCL__TCPClientWrapper_t {
    QByteArrayData data[7];
    char stringdata0[91];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_RW__CCL__TCPClientWrapper_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_RW__CCL__TCPClientWrapper_t qt_meta_stringdata_RW__CCL__TCPClientWrapper = {
    {
QT_MOC_LITERAL(0, 0, 25), // "RW::CCL::TCPClientWrapper"
QT_MOC_LITERAL(1, 26, 4), // "Stop"
QT_MOC_LITERAL(2, 31, 0), // ""
QT_MOC_LITERAL(3, 32, 20), // "SetConfigForPipeline"
QT_MOC_LITERAL(4, 53, 13), // "sSimpleConfig"
QT_MOC_LITERAL(5, 67, 15), // "stConfiguration"
QT_MOC_LITERAL(6, 83, 7) // "Process"

    },
    "RW::CCL::TCPClientWrapper\0Stop\0\0"
    "SetConfigForPipeline\0sSimpleConfig\0"
    "stConfiguration\0Process"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_RW__CCL__TCPClientWrapper[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       3,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       2,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    0,   29,    2, 0x06 /* Public */,
       3,    1,   30,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
       6,    0,   33,    2, 0x0a /* Public */,

 // signals: parameters
    QMetaType::Void,
    QMetaType::Void, 0x80000000 | 4,    5,

 // slots: parameters
    QMetaType::Void,

       0        // eod
};

void RW::CCL::TCPClientWrapper::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        TCPClientWrapper *_t = static_cast<TCPClientWrapper *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->Stop(); break;
        case 1: _t->SetConfigForPipeline((*reinterpret_cast< sSimpleConfig(*)>(_a[1]))); break;
        case 2: _t->Process(); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        void **func = reinterpret_cast<void **>(_a[1]);
        {
            typedef void (TCPClientWrapper::*_t)();
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&TCPClientWrapper::Stop)) {
                *result = 0;
                return;
            }
        }
        {
            typedef void (TCPClientWrapper::*_t)(sSimpleConfig );
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&TCPClientWrapper::SetConfigForPipeline)) {
                *result = 1;
                return;
            }
        }
    }
}

const QMetaObject RW::CCL::TCPClientWrapper::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_RW__CCL__TCPClientWrapper.data,
      qt_meta_data_RW__CCL__TCPClientWrapper,  qt_static_metacall, Q_NULLPTR, Q_NULLPTR}
};


const QMetaObject *RW::CCL::TCPClientWrapper::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *RW::CCL::TCPClientWrapper::qt_metacast(const char *_clname)
{
    if (!_clname) return Q_NULLPTR;
    if (!strcmp(_clname, qt_meta_stringdata_RW__CCL__TCPClientWrapper.stringdata0))
        return static_cast<void*>(const_cast< TCPClientWrapper*>(this));
    return QObject::qt_metacast(_clname);
}

int RW::CCL::TCPClientWrapper::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 3)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 3;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 3)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 3;
    }
    return _id;
}

// SIGNAL 0
void RW::CCL::TCPClientWrapper::Stop()
{
    QMetaObject::activate(this, &staticMetaObject, 0, Q_NULLPTR);
}

// SIGNAL 1
void RW::CCL::TCPClientWrapper::SetConfigForPipeline(sSimpleConfig _t1)
{
    void *_a[] = { Q_NULLPTR, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}
QT_END_MOC_NAMESPACE
