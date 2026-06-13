import sys
deps = ["torch", "transformers", "sentence_transformers"]
for d in deps:
    try:
        mod = __import__(d)
        ver = getattr(mod, "__version__", "unknown")
        print(f"{d}: OK ({ver})")
    except ImportError:
        print(f"{d}: NOT INSTALLED")
