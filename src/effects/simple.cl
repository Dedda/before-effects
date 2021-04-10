__kernel void invert(__global uchar* in, __global uchar* out)
{
    __private int pixel = get_global_id(0) * 4;
    for (__private char i = 0; i < 3; i++)
    {
        out[pixel + i] = 255 ^ in[pixel + i];
    }
    out[pixel + 3] = in[pixel + 3];
}

__kernel void greyscale(__global uchar* in, __global uchar* out)
{
    __private int pixel = get_global_id(0) * 4;
    __private int sum = in[pixel] + in[pixel + 1] + in[pixel + 2];
    __private uchar color = (uchar) (sum / 3);
    for (__private char i = 0; i < 3; i++)
    {
        out[pixel + i] = color;
    }
    out[pixel + 3] = in[pixel + 3];
}