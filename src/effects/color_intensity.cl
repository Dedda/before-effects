__kernel void color_intensity(__global uchar* in, __global uchar* out, __global float* intensities)
{
    __private int pixel = get_global_id(0) * 4;
    for (__private char i = 0; i < 4; i++)
    {
        __private float intensity = intensities[i] * 2.0F;
        out[pixel + i] = (uchar) ((float) in[pixel + i] * intensity);
    }
}