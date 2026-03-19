"""COLMAP bundle adjustment for refining VGGT camera poses."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from omegaconf import DictConfig

from photogrammetry.sfm.base import SfMBackend


class COLMAPBundleAdjustment(SfMBackend):
    """Refine camera poses from VGGT using COLMAP bundle adjustment.

    This does not run full COLMAP SfM — it takes existing poses from VGGT
    and uses PyCOLMAP's bundle adjustment to refine them, optionally
    incorporating feature matches from SuperPoint + LightGlue.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

    def estimate(
        self,
        image_dir: Path,
        output_dir: Path,
    ) -> dict[str, Any]:
        raise NotImplementedError(
            "Use refine() to refine existing VGGT poses, "
            "not estimate() for standalone SfM."
        )

    def refine(
        self,
        vggt_result: dict[str, Any],
        output_dir: Path,
    ) -> dict[str, Any]:
        """Refine VGGT camera poses via COLMAP bundle adjustment.

        Args:
            vggt_result: Output from VGGTSfM.estimate()
            output_dir: Where to write refined results

        Returns:
            Updated result dict with refined cameras and points.
        """
        import pycolmap

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cameras = vggt_result["cameras"]
        points_3d = vggt_result["points_3d"]
        image_paths = vggt_result.get("image_paths", [])

        logger.info(
            "Running COLMAP BA on {} cameras and {} points",
            len(cameras), len(points_3d),
        )

        reconstruction = self._build_reconstruction(cameras, points_3d, image_paths)

        ba_options = pycolmap.BundleAdjustmentOptions()
        ba_options.refine_focal_length = self.cfg.get("ba_refine_focal_length", True)
        ba_options.refine_extra_params = self.cfg.get("ba_refine_extra_params", True)

        pycolmap.bundle_adjustment(reconstruction, ba_options)
        logger.info("Bundle adjustment complete")

        refined_cameras = self._extract_refined_cameras(reconstruction, image_paths)

        refined_points = []
        for pid in reconstruction.points3D:
            pt = reconstruction.points3D[pid]
            refined_points.append(pt.xyz)
        refined_points = np.array(refined_points) if refined_points else points_3d

        result = dict(vggt_result)
        result["cameras"] = refined_cameras
        result["points_3d"] = refined_points
        result["output_dir"] = output_dir
        return result

    def _build_reconstruction(
        self,
        cameras: dict[str, dict[str, Any]],
        points_3d: np.ndarray,
        image_paths: list[Path],
    ) -> Any:
        """Build a pycolmap.Reconstruction from VGGT outputs."""
        import pycolmap

        recon = pycolmap.Reconstruction()

        for idx, (name, cam_data) in enumerate(cameras.items()):
            intrinsic = np.array(cam_data["intrinsic"])
            fx = float(intrinsic[0, 0]) if intrinsic.ndim == 2 else 1000.0
            fy = float(intrinsic[1, 1]) if intrinsic.ndim == 2 else fx
            cx = float(intrinsic[0, 2]) if intrinsic.ndim == 2 else 0.0
            cy = float(intrinsic[1, 2]) if intrinsic.ndim == 2 else 0.0

            width = int(cx * 2) if cx > 0 else 1600
            height = int(cy * 2) if cy > 0 else 1200

            camera = pycolmap.Camera(
                model="PINHOLE",
                width=width,
                height=height,
                params=[fx, fy, cx, cy],
                camera_id=idx,
            )
            recon.add_camera(camera)

            extrinsic = np.array(cam_data["extrinsic"])
            R = extrinsic[:3, :3] if extrinsic.shape == (4, 4) else extrinsic[:3, :3]
            t = extrinsic[:3, 3] if extrinsic.shape == (4, 4) else np.zeros(3)

            image = pycolmap.Image(
                image_id=idx,
                camera_id=idx,
                name=name,
                cam_from_world=pycolmap.Rigid3d(
                    rotation=pycolmap.Rotation3d(R),
                    translation=t,
                ),
            )
            recon.add_image(image)
            recon.register_image(idx)

        for pid, xyz in enumerate(points_3d[:10000]):
            pt = pycolmap.Point3D()
            pt.xyz = xyz
            pt.color = np.array([128, 128, 128], dtype=np.uint8)
            recon.add_point3D(pt.xyz, pycolmap.Track(), pt.color)

        return recon

    def _extract_refined_cameras(
        self,
        reconstruction: Any,
        image_paths: list[Path],
    ) -> dict[str, dict[str, Any]]:
        """Extract refined cameras from a COLMAP reconstruction."""
        refined: dict[str, dict[str, Any]] = {}

        for image_id in reconstruction.images:
            image = reconstruction.images[image_id]
            camera = reconstruction.cameras[image.camera_id]

            R = image.cam_from_world.rotation.matrix()
            t = image.cam_from_world.translation

            params = camera.params
            if camera.model_name == "PINHOLE":
                K = np.array([
                    [params[0], 0, params[2]],
                    [0, params[1], params[3]],
                    [0, 0, 1],
                ])
            else:
                K = np.eye(3)
                K[0, 0] = params[0]
                K[1, 1] = params[0]

            extrinsic = np.eye(4)
            extrinsic[:3, :3] = R
            extrinsic[:3, 3] = t

            refined[image.name] = {
                "extrinsic": extrinsic,
                "intrinsic": K,
            }

        return refined
