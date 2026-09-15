import threading

# Shared across STTManager and TTSManager: moshi's CUDAGraphed wrappers capture a
# new CUDA graph on the first step(s) of every streaming session. If another
# thread launches any GPU op on the same stream while a capture is in progress,
# CUDA raises "operation not permitted when stream is capturing". Serializing all
# GPU-touching calls through this lock avoids that race.
GPU_LOCK = threading.Lock()
