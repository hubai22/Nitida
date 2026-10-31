# Replaces the CodeFormer "weight" graph input with a constant (ONNX Runtime on iOS cannot feed double tensors).
# Usage: python3 bake_onnx_scalar.py in.onnx out.onnx 0.9
import sys, struct
src, dst, value = sys.argv[1], sys.argv[2], float(sys.argv[3])
data = open(src, 'rb').read()
def rv(b, i):
    r = s = 0
    while True:
        c = b[i]; i += 1; r |= (c & 0x7f) << s; s += 7
        if c < 0x80: return r, i
def ev(n):
    o = bytearray()
    while True:
        c = n & 0x7f; n >>= 7
        if n: o.append(c | 0x80)
        else: o.append(c); return bytes(o)
def fields(b):
    i = 0
    while i < len(b):
        st = i; key, i = rv(b, i); f, wt = key >> 3, key & 7
        if wt == 0: _, i = rv(b, i); yield f, wt, None, b[st:i]
        elif wt == 2:
            ln, i = rv(b, i); yield f, wt, b[i:i+ln], b[st:i+ln]; i += ln
        elif wt == 1: i += 8; yield f, wt, None, b[st:i]
        elif wt == 5: i += 4; yield f, wt, None, b[st:i]
        else: raise ValueError(wt)
def name_of(vi):
    for f, wt, p, _ in fields(vi):
        if f == 1 and wt == 2: return p.decode()
out = bytearray(); done = False
for f, wt, payload, raw in fields(data):
    if f == 7 and wt == 2:   # GraphProto
        g = bytearray(); removed = 0
        for gf, gwt, gp, graw in fields(payload):
            if gf == 11 and gwt == 2 and name_of(gp) == 'weight':
                removed += 1; continue
            g += graw
        # TensorProto: dims none (scalar), data_type=11 (DOUBLE) field 2, name field 8, raw_data field 9
        t = ev((2 << 3) | 0) + ev(11) + ev((8 << 3) | 2) + ev(6) + b'weight' + ev((9 << 3) | 2) + ev(8) + struct.pack('<d', value)
        g += ev((5 << 3) | 2) + ev(len(t)) + t
        out += ev((7 << 3) | 2) + ev(len(g)) + g
        done = removed == 1
    else:
        out += raw
assert done, 'weight input not found'
open(dst, 'wb').write(out)
print('ok', len(out))
