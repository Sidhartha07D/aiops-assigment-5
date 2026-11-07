# AI Usage Disclosure

## AI Tool Used

- OpenAI Codex (ChatGPT)

## How It Was Used and Impact

1. **Conceptual Clarification**
   - I used AI to discuss the assignment requirements and the ideas behind FastAPI validation and metrics, load testing and Little's Law, MLflow model versions and aliases, promotion gates, shadow testing, Ray Serve canaries, and release smoke tests.
   - **Impact:** This helped me connect the four parts and explain what the measured results mean.

2. **Code Generation**
   - AI helped draft and revise the implementation and tests in `partA/` through `partD/`: `eta_app.py`, `test_eta_app.py`, `eta_mlflow.py`, `test_gate.py`, `shadow_eta.py`, `eta_ray.py`, `verify_ray.py`, and `smoke_eta.py`. The supplied `eta_data.py` and `load_test.py` were starter files; the model bundles were generated from the starter script.
   - **Impact:** AI accelerated implementation, leaving more time to run the services, test behavior, measure performance, and inspect the outputs. The related DA3408 lecture notebooks are cited in `REPORT.md`.

3. **Debugging and Verification**
   - AI helped plan commands for my Ubuntu ARM64 VM, interpret test failures and library warnings, compare the one-worker and two-worker load results, and check MLflow, shadow, Ray Serve, and smoke-test outputs. I ran the commands in the VM and supplied the resulting terminal output and screenshots for review.
   - **Impact:** This helped me verify the behavior and keep the report's claims tied to actual runs. A clean extraction of the submission passed 29 tests, and I reran MLflow registration and the default promotion gate from it.

4. **Report, Documentation, and Video Assistance**
   - AI helped organize `REPORT.md` and `README.md`, calculate and present values from the saved result files, and plan a short video walkthrough. The saved outputs were used to check the reported values.
   - **Impact:** The submission records the methods, measured results, limitations, and reproduction steps clearly.

## Responsibility

- The reported measurements and screenshots came from real runs on my Ubuntu ARM64 VM (four CPU cores, 3.3 GiB RAM, Python 3.11.18); they were not invented by AI.
- I checked the final submission against the saved outputs and remain responsible for the code, claims, and explanations I submit.
- `REPORT.md` identifies the work that was not tested, including real production traffic, multi-node Ray, and a live rolling restart or model hot swap.
