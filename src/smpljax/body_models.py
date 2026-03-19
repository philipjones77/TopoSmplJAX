from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from .landmarks import find_dynamic_lmk_idx_and_bcoords, vertices2landmarks
from .lbs import batch_rodrigues, lbs
from .utils import SMPLModelData, as_jax_array, normalize_pose_input, zeros_pose


Array = jnp.ndarray


@dataclass(frozen=True)
class ModelOutput:
    vertices: Array
    joints: Array
    full_pose: Array | None = None


@dataclass(frozen=True)
class SMPLJAXModel:
    """Minimal callable SMPL-family model wrapper."""

    data: SMPLModelData

    def _num_body_joints(self) -> int:
        if self.data.num_body_joints is not None:
            return int(self.data.num_body_joints)
        total = int(self.data.parents.shape[0])
        return max(total - 1, 0)

    def export_template_mesh(self) -> dict[str, object]:
        from .mesh_export import export_template_mesh

        return export_template_mesh(self)

    def export_posed_mesh(self, params: dict[str, object]) -> dict[str, object]:
        from .mesh_export import export_posed_mesh

        return export_posed_mesh(self, params)

    def export_ct_mesh_payload_template(self) -> dict[str, object]:
        from .mesh_export import export_ct_mesh_payload_template

        return export_ct_mesh_payload_template(self)

    def export_ct_mesh_payload_pose(self, params: dict[str, object]) -> dict[str, object]:
        from .mesh_export import export_ct_mesh_payload_pose

        return export_ct_mesh_payload_pose(self, params)

    def to_randomfields77_static_domain_payload(self) -> dict[str, object]:
        from .mesh_export import to_randomfields77_static_domain_payload

        return to_randomfields77_static_domain_payload(self)

    def to_randomfields77_dynamic_mesh_state(self, params: dict[str, object]) -> dict[str, object]:
        from .mesh_export import to_randomfields77_dynamic_mesh_state

        return to_randomfields77_dynamic_mesh_state(self, params)

    def _compose_full_pose(
        self,
        body_pose: Array,
        global_orient: Array,
        jaw_pose: Array | None,
        leye_pose: Array | None,
        reye_pose: Array | None,
        left_hand_pose: Array | None,
        right_hand_pose: Array | None,
        pose2rot: bool,
    ) -> Array:
        def _default_pose(batch: int, joints: int, dtype: jnp.dtype) -> Array:
            if pose2rot:
                return zeros_pose(batch, joints, dtype)
            return jnp.broadcast_to(jnp.eye(3, dtype=dtype), (batch, joints, 3, 3))

        pieces = [global_orient, body_pose]
        batch = body_pose.shape[0]
        dtype = body_pose.dtype
        if int(self.data.num_face_joints) > 0:
            if int(self.data.num_face_joints) != 3:
                raise ValueError("Only 3 face joints are currently supported (jaw, leye, reye)")
            pieces.extend(
                [
                    jaw_pose if jaw_pose is not None else _default_pose(batch, 1, dtype),
                    leye_pose if leye_pose is not None else _default_pose(batch, 1, dtype),
                    reye_pose if reye_pose is not None else _default_pose(batch, 1, dtype),
                ]
            )
        if int(self.data.num_hand_joints) > 0:
            pieces.extend(
                [
                    left_hand_pose
                    if left_hand_pose is not None
                    else _default_pose(batch, int(self.data.num_hand_joints), dtype),
                    right_hand_pose
                    if right_hand_pose is not None
                    else _default_pose(batch, int(self.data.num_hand_joints), dtype),
                ]
            )
        full_pose = jnp.concatenate(pieces, axis=(-2 if pose2rot else -3))
        if pose2rot:
            if full_pose.ndim == 4 and full_pose.shape[-2:] == (3, 3):
                raise ValueError("pose2rot=True expects axis-angle pose tensors, not rotation matrices")
        else:
            if full_pose.ndim != 4 or full_pose.shape[-2:] != (3, 3):
                raise ValueError("pose2rot=False expects rotation-matrix pose tensors with shape (B, J, 3, 3)")
        if pose2rot and self.data.pose_mean is not None:
            pose_mean = as_jax_array(self.data.pose_mean, dtype=full_pose.dtype)
            if pose_mean.ndim == 1:
                pose_mean = pose_mean.reshape(-1, 3)
            full_pose = full_pose + pose_mean[None, :, :]
        return full_pose

    def _normalize_pose_component(self, pose: Array | None, pose2rot: bool) -> Array | None:
        if pose is None:
            return None
        pose = as_jax_array(pose)
        if pose2rot:
            return normalize_pose_input(pose)
        if pose.ndim == 4 and pose.shape[-2:] == (3, 3):
            return pose
        if pose.ndim == 3 and pose.shape[-1] == 9:
            return pose.reshape(pose.shape[0], pose.shape[1], 3, 3)
        if pose.ndim == 2 and (pose.shape[-1] % 9 == 0):
            return pose.reshape(pose.shape[0], pose.shape[-1] // 9, 3, 3)
        raise ValueError("pose2rot=False expects pose component in rotation-matrix form")

    def _batch_size(self, x: Array | None) -> int | None:
        if x is None:
            return None
        return int(x.shape[0])

    def _validate_batch_size(self, name: str, x: Array | None, expected_batch: int) -> None:
        batch = self._batch_size(x)
        if batch is not None and batch != expected_batch:
            raise ValueError(f"{name} batch size must match betas batch size {expected_batch}, got {batch}")

    def _validate_optional_inputs(
        self,
        *,
        betas: Array,
        expression: Array | None,
        jaw_pose: Array | None,
        leye_pose: Array | None,
        reye_pose: Array | None,
        left_hand_pose: Array | None,
        right_hand_pose: Array | None,
        transl: Array | None,
        pose_joint_axis: int,
    ) -> None:
        batch_size = int(betas.shape[0])
        self._validate_batch_size("expression", expression, batch_size)
        self._validate_batch_size("jaw_pose", jaw_pose, batch_size)
        self._validate_batch_size("leye_pose", leye_pose, batch_size)
        self._validate_batch_size("reye_pose", reye_pose, batch_size)
        self._validate_batch_size("left_hand_pose", left_hand_pose, batch_size)
        self._validate_batch_size("right_hand_pose", right_hand_pose, batch_size)
        self._validate_batch_size("transl", transl, batch_size)

        num_expr = int(self.data.num_expression_coeffs or 0)
        if expression is not None:
            if expression.ndim != 2:
                raise ValueError("expression must have shape (B, E)")
            if num_expr > 0 and expression.shape[1] > num_expr:
                raise ValueError(
                    f"expression must contain at most {num_expr} coefficients, got {expression.shape[1]}"
                )

        num_face = int(self.data.num_face_joints or 0)
        face_inputs = {"jaw_pose": jaw_pose, "leye_pose": leye_pose, "reye_pose": reye_pose}
        if num_face == 0 and any(x is not None for x in face_inputs.values()):
            raise ValueError("face pose inputs were provided but this model has no face joints")
        for name, pose in face_inputs.items():
            if pose is not None and pose.shape[pose_joint_axis] != 1:
                raise ValueError(f"{name} must contain exactly one joint")

        num_hand = int(self.data.num_hand_joints or 0)
        hand_inputs = {"left_hand_pose": left_hand_pose, "right_hand_pose": right_hand_pose}
        if num_hand == 0 and any(x is not None for x in hand_inputs.values()):
            raise ValueError("hand pose inputs were provided but this model has no hand joints")
        if num_hand > 0 and not self.data.use_pca:
            for name, pose in hand_inputs.items():
                if pose is not None and pose.shape[pose_joint_axis] != num_hand:
                    raise ValueError(f"{name} must contain {num_hand} joints, got {pose.shape[pose_joint_axis]}")

        if transl is not None and (transl.ndim != 2 or transl.shape[1] != 3):
            raise ValueError("transl must have shape (B, 3)")

        total_shape_coeffs = int(self.data.shapedirs.shape[-1])
        expr_dim = 0 if expression is None else int(expression.shape[1])
        if expression is not None and num_expr == 0 and int(betas.shape[1]) + expr_dim > total_shape_coeffs:
            raise ValueError("expression was provided but this model does not support expression coefficients")
        if int(betas.shape[1]) + expr_dim > total_shape_coeffs:
            raise ValueError(
                "betas + expression exceeds shapedirs capacity: "
                f"{int(betas.shape[1]) + expr_dim} > {total_shape_coeffs}"
            )

    def _apply_hand_pca_if_needed(
        self, hand_pose: Array | None, components: Array | None, pose2rot: bool
    ) -> Array | None:
        if hand_pose is None:
            return None
        hand_pose = as_jax_array(hand_pose)
        if not self.data.use_pca:
            return self._normalize_pose_component(hand_pose, pose2rot=pose2rot)
        if not pose2rot:
            raise ValueError("use_pca=True with pose2rot=False is not supported. Provide explicit rotation matrices.")
        if components is None:
            raise ValueError("use_pca=True but PCA components are missing")
        comps = as_jax_array(components, dtype=hand_pose.dtype)
        if hand_pose.ndim != 2:
            return self._normalize_pose_component(hand_pose, pose2rot=pose2rot)
        aa_flat = jnp.einsum("bp,pk->bk", hand_pose, comps)
        aa = aa_flat.reshape(hand_pose.shape[0], int(self.data.num_hand_joints), 3)
        if pose2rot:
            return aa
        return batch_rodrigues(aa.reshape(-1, 3)).reshape(hand_pose.shape[0], int(self.data.num_hand_joints), 3, 3)

    def _augment_and_map_joints(self, vertices: Array, joints: Array, full_pose: Array, pose2rot: bool) -> Array:
        if self.data.extra_vertex_ids:
            ids = jnp.asarray(self.data.extra_vertex_ids, dtype=jnp.int32)
            extra = vertices[..., ids, :]
            joints = jnp.concatenate([joints, extra], axis=-2)

        if self.data.faces_tensor is not None and self.data.lmk_faces_idx is not None and self.data.lmk_bary_coords is not None:
            lmk_faces_idx = as_jax_array(self.data.lmk_faces_idx, dtype=jnp.int32)[None, ...]
            lmk_faces_idx = jnp.repeat(lmk_faces_idx, vertices.shape[0], axis=0)
            lmk_bary_coords = as_jax_array(self.data.lmk_bary_coords, dtype=vertices.dtype)[None, ...]
            lmk_bary_coords = jnp.repeat(lmk_bary_coords, vertices.shape[0], axis=0)

            if (
                self.data.use_face_contour
                and self.data.dynamic_lmk_faces_idx is not None
                and self.data.dynamic_lmk_bary_coords is not None
                and self.data.neck_kin_chain is not None
            ):
                dyn_faces_idx, dyn_bary = find_dynamic_lmk_idx_and_bcoords(
                    vertices=vertices,
                    pose=full_pose,
                    dynamic_lmk_faces_idx=as_jax_array(self.data.dynamic_lmk_faces_idx, dtype=jnp.int32),
                    dynamic_lmk_bary_coords=as_jax_array(self.data.dynamic_lmk_bary_coords, dtype=vertices.dtype),
                    neck_kin_chain=as_jax_array(self.data.neck_kin_chain, dtype=jnp.int32),
                    pose2rot=pose2rot,
                )
                lmk_faces_idx = jnp.concatenate([lmk_faces_idx, dyn_faces_idx], axis=1)
                lmk_bary_coords = jnp.concatenate([lmk_bary_coords, dyn_bary], axis=1)

            landmarks = vertices2landmarks(
                vertices=vertices,
                faces=as_jax_array(self.data.faces_tensor, dtype=jnp.int32),
                lmk_faces_idx=lmk_faces_idx,
                lmk_bary_coords=lmk_bary_coords,
            )
            joints = jnp.concatenate([joints, landmarks], axis=-2)

        if self.data.joint_mapper is not None:
            joints = self.data.joint_mapper(joints)
        return joints

    def __call__(
        self,
        betas: Array,
        body_pose: Array,
        global_orient: Array | None = None,
        transl: Array | None = None,
        expression: Array | None = None,
        jaw_pose: Array | None = None,
        leye_pose: Array | None = None,
        reye_pose: Array | None = None,
        left_hand_pose: Array | None = None,
        right_hand_pose: Array | None = None,
        return_full_pose: bool = False,
        pose2rot: bool = True,
    ) -> ModelOutput:
        pose_joint_axis = -2 if pose2rot else -3
        betas = as_jax_array(betas)
        if betas.ndim != 2:
            raise ValueError("betas must have shape (B, num_betas)")
        body_pose = self._normalize_pose_component(body_pose, pose2rot=pose2rot)
        if body_pose is None:
            raise ValueError("body_pose cannot be None")
        expected_body_joints = self._num_body_joints()
        if body_pose.shape[pose_joint_axis] != expected_body_joints:
            raise ValueError(
                f"body_pose must contain {expected_body_joints} joints, got {body_pose.shape[pose_joint_axis]}"
            )
        if global_orient is None:
            if pose2rot:
                global_orient = jnp.zeros((betas.shape[0], 1, 3), dtype=betas.dtype)
            else:
                global_orient = jnp.broadcast_to(jnp.eye(3, dtype=betas.dtype), (betas.shape[0], 1, 3, 3))
        else:
            global_orient = self._normalize_pose_component(global_orient, pose2rot=pose2rot)
            if global_orient.shape[pose_joint_axis] != 1:
                raise ValueError("global_orient must contain exactly one joint")

        if expression is not None:
            expression = as_jax_array(expression, dtype=betas.dtype)

        jaw_pose = self._normalize_pose_component(jaw_pose, pose2rot=pose2rot)
        leye_pose = self._normalize_pose_component(leye_pose, pose2rot=pose2rot)
        reye_pose = self._normalize_pose_component(reye_pose, pose2rot=pose2rot)
        left_hand_pose = self._apply_hand_pca_if_needed(
            left_hand_pose, self.data.left_hand_components, pose2rot=pose2rot
        )
        right_hand_pose = self._apply_hand_pca_if_needed(
            right_hand_pose, self.data.right_hand_components, pose2rot=pose2rot
        )
        transl = as_jax_array(transl, dtype=betas.dtype) if transl is not None else None

        self._validate_optional_inputs(
            betas=betas,
            expression=expression,
            jaw_pose=jaw_pose,
            leye_pose=leye_pose,
            reye_pose=reye_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            transl=transl,
            pose_joint_axis=pose_joint_axis,
        )
        shape_components = jnp.concatenate([betas, expression], axis=-1) if expression is not None else betas

        full_pose = self._compose_full_pose(
            body_pose=body_pose,
            global_orient=global_orient,
            jaw_pose=jaw_pose,
            leye_pose=leye_pose,
            reye_pose=reye_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            pose2rot=pose2rot,
        )
        shapedirs = self.data.shapedirs[..., : shape_components.shape[-1]]
        verts, joints = lbs(
            betas=shape_components,
            pose=full_pose,
            v_template=self.data.v_template,
            shapedirs=shapedirs,
            posedirs=self.data.posedirs,
            j_regressor=self.data.j_regressor,
            parents=self.data.parents,
            lbs_weights=self.data.lbs_weights,
            pose2rot=pose2rot,
        )
        if transl is not None:
            verts = verts + transl[:, None, :]
            joints = joints + transl[:, None, :]
        joints = self._augment_and_map_joints(vertices=verts, joints=joints, full_pose=full_pose, pose2rot=pose2rot)
        return ModelOutput(vertices=verts, joints=joints, full_pose=full_pose if return_full_pose else None)
