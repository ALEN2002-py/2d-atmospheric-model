FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so they're cached across source-only rebuilds
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY tests/test_grid.py tests/test_integrators.py tests/
COPY run_model.py compare_schemes.py plot_results.py run_all.py ./

# Sanity check: prove the numerics work out of the box, at build time.
# Scoped to the two test files this minimal image actually has the code
# for -- tests/test_api.py and tests/test_ml_surrogate.py need api/ and
# ml/ (see Dockerfile.api), which this CLI-only image doesn't include.
RUN pytest tests/ -q

ENTRYPOINT ["python", "run_model.py"]
CMD ["--scheme", "RK4", "--bubble_amp", "2.0", "--dt", "0.02", "--n_steps", "500"]
