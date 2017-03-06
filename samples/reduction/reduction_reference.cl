void atomic_add_global(volatile global float *source, const float operand) {
    union {
        unsigned int intVal;
        float floatVal;
    } newVal;
    union {
        unsigned int intVal;
        float floatVal;
    } prevVal;

    do {
        prevVal.floatVal = *source;
        newVal.floatVal = prevVal.floatVal + operand;
    } while (atomic_cmpxchg((volatile global unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}


__kernel void reduceReference(__global const float* in, __global float* out, unsigned int n) {
    unsigned int tid = get_local_id(0);
    unsigned int i = get_global_id(0);

    //XXX hard optimization for 256-thread work groups
    __local float buf[256];
    if (i < n)
        buf[tid] = in[i];
    else
        buf[tid] = 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 128)
        buf[tid] += buf[tid + 128];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 64)
        buf[tid] += buf[tid + 64];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 32)
        buf[tid] += buf[tid + 32];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 16)
        buf[tid] += buf[tid + 16];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 8)
        buf[tid] += buf[tid + 8];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 4)
        buf[tid] += buf[tid + 4];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 2)
        buf[tid] += buf[tid + 2];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 1) {
        buf[0] += buf[1];
        atomic_add_global(out, buf[0]);
    }
} 


