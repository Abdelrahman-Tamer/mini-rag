import py_compile, sys

files = [
    'src/helpers/config.py',
    'src/routes/schemes/data.py',
    'src/controllers/NLPController.py',
    'src/stores/llm/templates/locales/en/rag.py',
    'src/stores/llm/templates/locales/ar/rag.py',
    'src/stores/rerank/RerankInterface.py',
    'src/stores/rerank/RerankEnums.py',
    'src/stores/rerank/RerankProviderFactory.py',
    'src/stores/rerank/providers/OllamaRerankProvider.py',
    'src/routes/nlp.py',
    'src/routes/data.py',
    'src/main.py',
]

errors = []
for f in files:
    try:
        py_compile.compile(f, doraise=True)
        print(f'OK: {f}')
    except py_compile.PyCompileError as e:
        errors.append(f)
        print(f'FAIL: {f} -> {e}')

if errors:
    print(f'\n{len(errors)} file(s) failed')
    sys.exit(1)
else:
    print('\nAll files compiled successfully')
