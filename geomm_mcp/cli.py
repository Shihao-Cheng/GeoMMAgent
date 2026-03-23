# geomm_mcp/cli.py
# Run: ``python -m geomm_mcp.cli --toolkit classification`` from the GeoMMAgent repo root.

from __future__ import annotations

import argparse
import os


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _build_toolkit(name: str):
    name = name.lower().strip()
    if name == "classification":
        from toolkit.classification_toolkit import ClassificationToolkit

        p = os.environ.get("GEOMM_CLS_YOLO_PATH", "").strip()
        return ClassificationToolkit(yolo_model_path=p) if p else ClassificationToolkit()
    if name == "detection":
        from toolkit.detection_toolkit import YOLODetectionToolkit

        p = os.environ.get("GEOMM_DET_YOLO_PATH", "").strip()
        return YOLODetectionToolkit(yolo_model_path=p) if p else YOLODetectionToolkit()
    if name == "segmentation":
        from toolkit.segmentation_toolkit import SegmentationToolkit

        root = _repo_root()
        ckpt = (
            os.environ.get("GEOMM_DEEPLAB_WEIGHTS", "").strip()
            or os.path.join(root, "weights", "deeplabv3plus-loveda.pth")
        )
        nc = int(os.environ.get("GEOMM_DEEPLAB_NUM_CLASSES", "7"))
        os_stride = int(os.environ.get("GEOMM_DEEPLAB_OUTPUT_STRIDE", "16"))
        return SegmentationToolkit(
            checkpoint_path=ckpt,
            num_classes=nc,
            output_stride=os_stride,
        )
    if name == "knowledge":
        from toolkit.knowledge_toolkit import KnowledgeToolkit

        return KnowledgeToolkit()
    if name == "general":
        from toolkit.general_toolkit import GeneralToolkit

        return GeneralToolkit()
    if name == "search":
        from toolkit.search_toolkit import SearchRSToolkit

        return SearchRSToolkit()
    raise ValueError(f"unknown toolkit: {name!r}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Start a FastMCP server for a GeoMMAgent toolkit (CAMEL @MCPServer). "
            "Run from the repository root with PYTHONPATH including this project."
        )
    )
    parser.add_argument(
        "--toolkit",
        required=True,
        choices=[
            "classification",
            "detection",
            "segmentation",
            "knowledge",
            "general",
            "search",
        ],
        help="Which toolkit to expose.",
    )
    parser.add_argument(
        "--transport",
        default="stdio",
        choices=["stdio", "sse", "streamable-http"],
        help="MCP transport (default: stdio for local MCP clients).",
    )
    args = parser.parse_args(argv)

    tk = _build_toolkit(args.toolkit)
    tk.mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
