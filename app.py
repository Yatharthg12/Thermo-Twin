"""WSGI entry point for local ThermoTwin service hosts."""

from pathlib import Path
import subprocess
import sys


project_root = Path(__file__).resolve().parent

# People naturally try ``python app.py``. If the prepared project environment
# exists, use its interpreter so model-library versions match the registry.
if __name__ == "__main__":
    candidates = (
        project_root / ".venv" / "Scripts" / "python.exe",
        project_root / ".venv" / "bin" / "python",
    )
    project_python = next((path for path in candidates if path.is_file()), None)
    if project_python is not None and project_python.resolve() != Path(sys.executable).resolve():
        child = subprocess.Popen([str(project_python), str(Path(__file__).resolve()), *sys.argv[1:]])
        try:
            exit_code = child.wait()
        except KeyboardInterrupt:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            exit_code = 130
        raise SystemExit(exit_code)

# Support the intuitive ``python app.py`` development command from a source
# checkout. Installed/packaged launches continue to resolve the package normally.
source_root = project_root / "src"
if str(source_root) not in sys.path:
    sys.path.insert(0, str(source_root))

from thermotwin.web import create_app

app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
