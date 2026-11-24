// SPDX-License-Identifier: MIT
// m2v_probe's M2V_RGB10A2 draw: xcrun -sdk iphoneos metal -std=ios-metal2.4 -o half.metallib half.metal (to /var/root/half.metallib)
#include <metal_stdlib>
using namespace metal;
// a quad over the top half of the target (NDC y from 0 to 1), solid colour
vertex float4 half_vs(uint v [[vertex_id]]) {
    float2 p[6] = { float2(-1, 0), float2(1, 0), float2(-1, 1), float2(1, 0), float2(1, 1), float2(-1, 1) };
    return float4(p[v], 0, 1);
}
fragment half4 half_fs() { return half4(0, 1, 0, 1); }
