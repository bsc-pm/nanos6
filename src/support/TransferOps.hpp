#ifndef TRANSFER_OPS_HPP
#define TRANSFER_OPS_HPP

#include <atomic>
#include "../lowlevel/SpinLock.hpp"

class TransferOps {
private:
    std::atomic<unsigned int> _pendingTransferOps;
    SpinLock _pendingCacheOp;
    ///*debug:*/ int _owner;
    ///*debug:*/ WorkDescriptor const *_wd;
    ///*debug:*/ int _loc;
public:
    TransferOps();
    ~TransferOps();
    void completeOp(){ _pendingTransferOps--; }
    void addOp() { _pendingTransferOps++; }
    bool allCompleted() { return _pendingTransferOps.load() == 0; }

    //bool addCacheOp( /* debug: */ WorkDescriptor const *wd, int loc = -1 );
    //void completeCacheOp( /* debug: */WorkDescriptor const *wd );
    //bool allCacheOpsCompleted();
};

#endif //TRANSFER_OPS_HPP
