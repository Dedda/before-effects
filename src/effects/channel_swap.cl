__kernel void channel_swap(__global uchar* in, __global uchar* out, __global int* order)
{
    __private int pixel = get_global_id(0) * 4;
    for (__private char i = 0; i < 4; i++)
    {
        out[pixel + i] = in[pixel + order[i]];
    }
}