"""Tests for the CLI argument parser and entry point."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tatvton.cli import build_parser, main, _parse_point, _parse_bbox


# ── Parser helpers ────────────────────────────────────────────────────


class TestParsePoint:
    def test_valid(self):
        assert _parse_point("256,384") == (256, 384)

    def test_negative(self):
        assert _parse_point("-10,20") == (-10, 20)

    def test_too_few_parts(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["b.jpg", "t.png", "--point", "123"])

    def test_non_integer(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["b.jpg", "t.png", "--point", "1.5,2.5"])


class TestParseBbox:
    def test_valid(self):
        assert _parse_bbox("10,20,300,400") == (10, 20, 300, 400)

    def test_too_few_parts(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["b.jpg", "t.png", "--bbox", "1,2,3"])

    def test_non_integer(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["b.jpg", "t.png", "--bbox", "a,b,c,d"])


# ── Argument parsing ─────────────────────────────────────────────────


class TestBuildParser:
    def test_point_prompt(self):
        args = build_parser().parse_args(["body.jpg", "tattoo.png", "--point", "100,200"])
        assert args.points == [(100, 200)]
        assert args.bbox is None

    def test_multiple_points(self):
        args = build_parser().parse_args([
            "b.jpg", "t.png", "--point", "10,20", "--point", "30,40",
        ])
        assert args.points == [(10, 20), (30, 40)]

    def test_bbox_prompt(self):
        args = build_parser().parse_args(["b.jpg", "t.png", "--bbox", "10,20,300,400"])
        assert args.bbox == (10, 20, 300, 400)
        assert args.points is None

    def test_region_required(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["b.jpg", "t.png"])

    def test_point_and_bbox_mutually_exclusive(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args([
                "b.jpg", "t.png", "--point", "1,2", "--bbox", "1,2,3,4",
            ])

    def test_defaults(self):
        args = build_parser().parse_args(["b.jpg", "t.png", "--point", "0,0"])
        assert str(args.output) == "output.png"
        assert args.resolution is None
        assert args.steps is None
        assert args.strength is None
        assert args.seed is None
        assert args.device is None
        assert args.offload is None
        assert args.save_mask is False
        assert args.save_raw is False

    def test_all_overrides(self):
        args = build_parser().parse_args([
            "b.jpg", "t.png", "--point", "50,50",
            "-o", "out.png",
            "--resolution", "512",
            "--steps", "20",
            "--strength", "0.9",
            "--guidance-scale", "5.0",
            "--ip-adapter-scale", "0.8",
            "--seed", "42",
            "--device", "cpu",
            "--offload", "sequential",
            "--save-mask",
            "--save-raw",
        ])
        assert args.resolution == 512
        assert args.steps == 20
        assert args.strength == 0.9
        assert args.guidance_scale == 5.0
        assert args.ip_adapter_scale == 0.8
        assert args.seed == 42
        assert args.device == "cpu"
        assert args.offload == "sequential"
        assert args.save_mask is True
        assert args.save_raw is True

    def test_invalid_device(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["b.jpg", "t.png", "--point", "0,0", "--device", "tpu"])

    def test_invalid_offload(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["b.jpg", "t.png", "--point", "0,0", "--offload", "bogus"])


# ── main() integration ───────────────────────────────────────────────


class TestMain:
    def test_missing_body_file(self, tmp_path: Path):
        """main() should error when body image doesn't exist."""
        tattoo = tmp_path / "tattoo.png"
        tattoo.write_bytes(b"fake")

        with pytest.raises(SystemExit):
            main([str(tmp_path / "nope.jpg"), str(tattoo), "--point", "0,0"])

    def test_missing_tattoo_file(self, tmp_path: Path):
        """main() should error when tattoo image doesn't exist."""
        body = tmp_path / "body.jpg"
        body.write_bytes(b"fake")

        with pytest.raises(SystemExit):
            main([str(body), str(tmp_path / "nope.png"), "--point", "0,0"])
