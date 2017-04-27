#include "KernelLevelThread.hpp"


__thread KernelLevelThread *KernelLevelThread::_currentKernelLevelThread(nullptr);
