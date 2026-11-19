import sys, importlib

print('exe :', sys.executable)
print('py  :', sys.version.split()[0])
print()

for m in ('numpy', 'scipy', 'torch'):
    try:
        mod = importlib.import_module(m)
        print('%-6s %s' % (m, getattr(mod, '__version__', '?')))
    except Exception as e:
        print('%-6s MISSING (%s)' % (m, type(e).__name__))

print()
try:
    import torch
    print('torch.version.cuda :', torch.version.cuda)
    print('torch.cuda.is_available :', torch.cuda.is_available())
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print('device : %s  %.1f GB  cc=%d.%d  sm=%d' % (
            p.name, p.total_memory / 1e9, p.major, p.minor, p.multi_processor_count))
        x = torch.randn(1024, 1024, device='cuda')
        y = x @ x
        torch.cuda.synchronize()
        print('cuda matmul smoke : OK', tuple(y.shape))
except Exception as e:
    print('torch cuda probe failed:', repr(e))
