# LBS Math Notes (Version 1.0.0)

The SMPL-family forward pass is built on linear blend skinning:
1. Shape deformation using `betas` and shape bases.
2. Pose correction using joint rotations and pose bases.
3. Kinematic chain transforms via parent tree.
4. Weighted skinning over per-vertex blend weights.

smplJAX implementation goals:
- preserve differentiability end-to-end;
- keep runtime JIT-compatible;
- control compile and memory behavior through explicit policy.
