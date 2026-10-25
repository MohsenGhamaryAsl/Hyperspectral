# Contributing

Thanks for taking the time to improve **Hyperspectral**!

## Reporting a problem

Open an [issue](https://github.com/MohsenGhamaryAsl/Hyperspectral/issues) and
include:

* what you did (the exact command or the code snippet),
* what you expected and what happened instead,
* your Python version and the versions of NumPy, SciPy and Matplotlib,
* the shape/dtype of the data cube you used if the problem is data related.

## Development setup

```bash
git clone https://github.com/MohsenGhamaryAsl/Hyperspectral.git
cd Hyperspectral
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

## Adding a feature

1. Keep the algorithms in `hyperspectral/`, the runnable demonstrations in
   `examples/` and the tests in `tests/`.
2. Every public function needs a NumPy-style docstring with parameters, return
   values and at least one small example.
3. Add a test that fails without your change. Synthetic data (see
   `tests/conftest.py`) keeps the suite fast - do not depend on the large data
   files in the tests.
4. Run the suite before pushing:

   ```bash
   MPLBACKEND=Agg pytest
   ```

5. Update `CHANGELOG.md` under the "Unreleased" heading.

## Style

* PEP 8, four-space indentation, maximum line length ~100 characters.
* Type hints on public functions.
* No plotting or printing inside the algorithms - return values instead, so the
  functions stay usable head-less and testable. Figures live in
  `hyperspectral/plotting.py`, messages in the CLI.

## Pull requests

* One topic per pull request, a short description and the motivation.
* The CI workflow runs the tests on Linux, macOS and Windows for Python 3.9,
  3.11 and 3.12; all jobs must be green.
* By contributing you agree that your work is released under the MIT licence of
  this project.
