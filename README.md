# GPU-Accelerated SAG Scheduling Analysis

Runtime source for the CUDA SAG solver.

## Build

```bash
# Main GPU solver
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j
cmake -S framework -B framework/build -DCMAKE_BUILD_TYPE=Release
cmake --build framework/build -j
```

## Run

```bash
# Main GPU solver
./build/expand_test -m <cores> <jobs.csv> <jobsprec.csv>

# Build helper for the main GPU solver
./run.sh -m <cores> <jobs.csv> <jobsprec.csv>
```

The GPU solver writes `jobs.rta.csv` beside the input `jobs.csv`.

## Input Files

`jobs.csv`:

```csv
Task ID, Job ID, Arrival min, Arrival max, Cost min, Cost max, Deadline, Priority
0, 0, 0, 0, 2, 5, 100, 1
```

`jobsprec.csv`:

```csv
Predecessor TID, Predecessor JID, Successor TID, Successor JID, sus_min, sus_max, type
```

Use a header-only `jobsprec.csv` if there are no precedence constraints.
