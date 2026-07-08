#include "Driver/GPU/AscendrtApi.h"
#include "Driver/Dispatch.h"

namespace proton {

namespace ascend {

struct ExternLibAscendrt : public ExternLibBase {
  using RetType = rtError_t;
  static constexpr const char *name = "libruntime.so";
  static constexpr const char *defaultDir = "";
  static constexpr RetType success = RT_ERROR_NONE;
  static void *lib;
};

void *ExternLibAscendrt::lib = nullptr;

DEFINE_DISPATCH(ExternLibAscendrt, ctxGetCurrent, rtCtxGetCurrent, rtContext_t *)

DEFINE_DISPATCH(ExternLibAscendrt, ctxGetDevice, rtGetDevice, int32_t *)

DEFINE_DISPATCH(ExternLibAscendrt, ctxGetStreamPriorityRange,
                rtDeviceGetStreamPriorityRange, int32_t *, int32_t *)

DEFINE_DISPATCH(ExternLibAscendrt, deviceGet, rtGetDeviceIndexByPhyId, uint32_t, uint32_t *)

DEFINE_DISPATCH(ExternLibAscendrt, streamCreateWithPriority, rtStreamCreate, rtStream_t *, int32_t)

DEFINE_DISPATCH(ExternLibAscendrt, streamSynchronize, rtStreamSynchronize, rtStream_t)

DEFINE_DISPATCH(ExternLibAscendrt, memcpyDToHAsync, rtMemcpyAsync, void *, uint64_t,
                void *, uint64_t, rtMemcpyKind_t, rtStream_t)

Device getDevice(uint64_t index) {
  return Device(DeviceType::ASCEND, index, 0 /*clockRate*/, 0 /*memoryClockRate*/, 0 /*busWidth*/,
                0 /*numSms*/, "" /*arch*/);
}

} // namespace ascend

} // namespace proton
