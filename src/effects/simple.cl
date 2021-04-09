__kernel void invert(__global uchar* in, __global uchar* out)
{
    int pixel = get_global_id(0) * 4;
    for (char i = 0; i < 3; i++)
    {
        out[pixel + i] = 255 - in[pixel + i];
    }
    out[pixel + 3] = in[pixel + 3];
}