# NMPDE problem suite

Each directory is self-contained: it has its own solver, example, CMake file
and, when needed, test mesh. There are no source-code dependencies between
problems, so one directory can be copied and submitted on its own.

| Directory | Use it for |
| --- | --- |
| `steady-adr` | Poisson, diffusion-reaction and steady advection-diffusion-reaction |
| `transient-adr` | Heat and transient advection-diffusion-reaction |
| `nonlinear-diffusion` | Nonlinear scalar diffusion solved with Newton's method |
| `linear-elasticity` | Vector-valued linear elasticity |
| `stokes` | Mixed velocity-pressure Stokes problems |
| `navier-stokes` | Steady Navier-Stokes with Picard linearization |
| `domain-decomposition` | Dirichlet-Neumann domain decomposition |

The original `lab-*` directories are unchanged.

## Build one problem

From inside the container opened with `nmpde`:

```bash
cd /home/ubuntu/Projects/nmpde-labs-aa-25-26/problem-suite/steady-adr
cmake -S . -B build
cmake --build build -j2
cd build
./exercise
```

For a new assignment, copy the closest directory and mainly edit
`exercise.cpp`: coefficients, forcing term, boundary conditions, mesh and
parameters. Change the solver files only if the weak formulation itself is
different.
