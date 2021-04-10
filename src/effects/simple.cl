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

__kernel void contrast(__global uchar* in, __global uchar* out, __global float* intensity)
{
    __private int pixel = get_global_id(0) * 4;
    for (__private char i  = 0; i < 3; i++)
    {
        __private float adjusted = ((float) in[pixel + i]) * 2;
        adjusted = adjusted / 255.0;
        adjusted = pow(adjusted, intensity[0] * 2);
        adjusted = adjusted * 255 / 2;
        adjusted = clamp(adjusted, 0.0F, 255.0F);
        
        out[pixel + i] = (uchar) adjusted;        
    }
    out[pixel + 3] = in[pixel + 3];
}

__kernel void brightness(__global uchar* in, __global uchar* out, __global float* intensity)
{
    __private int pixel = get_global_id(0) * 4;
    __private float factor = intensity[0] * 2;
    for (__private char i  = 0; i < 3; i++)
    {
        __private float adjusted = ((float) in[pixel + i]) * factor;
        out[pixel + i] = (uchar) clamp(adjusted, 0.0F, 255.0F);
    }
    out[pixel + 3] = in[pixel + 3];
}