import nox
import nox_uv

nox.options.sessions = ["tests", "docs"]
nox.options.default_venv_backend = "uv"


@nox_uv.session(reuse_venv=True, uv_groups=["test"])
def tests(session: nox.Session):
    """Run tests using test dependency group."""
    session.run("pytest")


@nox_uv.session(reuse_venv=True, uv_groups=["docs"])
def docs(session: nox.Session):
    """Build docs"""
    session.run("sphinx-autobuild", "--open-browser", "docs", "docs/_build/html")


@nox_uv.session(reuse_venv=True, uv_groups=["dev"])
def typecheck(session: nox.Session):
    """Run typecheck"""
    session.run("mypy", "src")
