import nox

nox.options.sessions = ['tests', 'docs']

def tests(session):
    """Run tests using pytest"""
    session.install('pytest')
    session.run('pytest')

def docs(session):
    """Build docs"""
    session.install('sphinx', 'myst-parser', 'sphinx-markdown-builder', 'furo')
    session.run('sphinx-build', 'docs', 'docs/_build/html')

def typecheck(session):
    """Run typecheck"""
    session.install("mypy")
    session.run("mypy", "src")