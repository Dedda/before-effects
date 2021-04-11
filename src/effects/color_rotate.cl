#define E .0000001f

bool fEqual(float x, float y)
{
    return (x+E > y && x-E < y);
}

float3 rgb2hsl(float3 values)
{
    // thread index and total
    float3 gMem = values;

    barrier(CLK_LOCAL_MEM_FENCE);

    gMem /= 255.0f; //convert from 256 color to float

    //calculate chroma
    float M = fmax(gMem.r, gMem.g);
    M = fmax(M, gMem.b);
    float m = fmin(gMem.r, gMem.g);
    m = fmin(m, gMem.b);
    float chroma = M-m; //calculate chroma
    float lightness = (M+m)/2.0f;
    float saturation = chroma/(1.0f-fabs(2.0f*lightness-1.0f));

    float hue = 0.0f;
    if (fEqual(gMem.r, M)) {
        hue = fmod((gMem.g - gMem.b)/chroma, 6.0f);
    } else if (fEqual(gMem.g, M)) {
        hue = (((gMem.b - gMem.r))/chroma) + 2.0f;
    } else {
        hue = (((gMem.r - gMem.g))/chroma) + 4.0f;
    }
    hue *= 60.0f;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (M == m)
        hue = saturation = 0;

    barrier(CLK_GLOBAL_MEM_FENCE);

    return (float3) (hue, saturation, lightness);
}


float3 hsl2rgb(float3 values)
{
    // thread index and total    
    float3 gMem = values;

    barrier(CLK_LOCAL_MEM_FENCE);

    float3 rgb = (float3)(0,0,0);

    //calculate chroma
    float chroma = (1.0f - fabs( (float)(2.0f*gMem.z - 1.0f) )) * gMem.y;
    float H = gMem.x/60.0f;
    float x = chroma * (1.0f - fabs( fmod(H, 2.0f) - 1.0f ));

    switch((int)H)
    {
        case 0:
            rgb = (float3)(chroma, x, 0);
            break;
        case 1:
            rgb = (float3)(x, chroma, 0);
            break;
        case 2:
            rgb = (float3)(0, chroma, x);
            break;
        case 3:
            rgb = (float3)(0, x, chroma);
            break;
        case 4:
            rgb = (float3)(x, 0, chroma);
            break;
        case 5:
            rgb = (float3)(chroma, 0, x);
            break;
        default:
            rgb = (float3)(0, 0, 0);    
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    rgb += gMem.z - .5f*chroma;
    rgb *= 255;

    return rgb;
}

__kernel void color_rotate(__global uchar* in, __global uchar* out, __global float* degrees)
{    
    int pixel = get_global_id(0) * 4;
    float3 rgb = (float3) (
        in[pixel], 
        in[pixel + 1], 
        in[pixel + 2]
    );

    float3 hsl = rgb2hsl(rgb);
    hsl.x += degrees[0];
    hsl.x = fmod(hsl.x, 360.0f);
    rgb = hsl2rgb(hsl);

    out[pixel]     = (uchar) rgb.x;
    out[pixel + 1] = (uchar) rgb.y;
    out[pixel + 2] = (uchar) rgb.z;
    out[pixel + 3] = in[pixel + 3];
}


__kernel void color_rotate_absolute(__global uchar* in, __global uchar* out, __global float* degrees)
{    
    int pixel = get_global_id(0) * 4;
    float3 rgb = (float3) (
        in[pixel], 
        in[pixel + 1], 
        in[pixel + 2]
    );

    float3 hsl = rgb2hsl(rgb);
    hsl.x = degrees[0];
    rgb = hsl2rgb(hsl);

    out[pixel]     = (uchar) rgb.x;
    out[pixel + 1] = (uchar) rgb.y;
    out[pixel + 2] = (uchar) rgb.z;
    out[pixel + 3] = in[pixel + 3];
}