# Mesh Movement Modes Theory

This note explains the mathematical difference between the five mesh movement modes.

## Core Distinction

The central split is between geometry variables and connectivity variables.

- Geometry variables are continuous node coordinates, deformation parameters, or boundary control values.
- Connectivity variables are combinatorial choices such as element incidence, edge flips, splits, collapses, and retriangulation events.

Automatic differentiation applies cleanly to continuous geometry maps. It does not apply cleanly to exact connectivity updates because the mapping from coordinates to topology is piecewise constant with discontinuous transitions.

## Mode 1: Fixed Topology AD

Connectivity is frozen. Geometry moves continuously. This is the cleanest AD setting and the baseline mathematical model for mesh motion in TopoJAX.

## Mode 2: Remesh Restart

Optimization is broken into phases. Each phase keeps topology fixed and differentiable. Between phases, a discrete remeshing operator changes connectivity. The composite process is not globally differentiable, but each phase is.

## Mode 3: Soft Connectivity Surrogate

A fixed candidate graph is chosen ahead of time. Instead of exact combinatorial choices, the method optimizes continuous weights over those candidates. This produces a differentiable surrogate objective but not exact topological dynamics.

## Mode 4: Straight-Through Connectivity

The forward pass uses hard discrete choices while the backward pass borrows gradients from a smooth surrogate. This is useful for experimentation, but the gradient is heuristic rather than exact.

## Mode 5: Fully Dynamic Remeshing

This aspirational mode would allow full remeshing while geometry is moving in a single optimization process. In exact form, this remains mathematically difficult because remeshing changes the optimization state dimension, incidence relations, and event structure. Any future realization will likely require relaxation, measure-valued formulations, or another surrogate mechanism rather than naive reverse-mode through exact remeshing.
