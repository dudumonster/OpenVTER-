from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from scripts.overlay_processed_obb_on_video import (
    TrailRenderer,
    build_raw_frame_tracks,
    render_trails,
    source_frame_for_base,
    write_video_clip,
)


def _track(track_id, x, y, class_name="car"):
    return {
        "track_id": str(track_id),
        "class_name": class_name,
        "center": np.asarray([x, y], dtype=float),
    }


def _renderer(mode, *, fps=10.0, seconds=1.0, max_gap=3):
    return TrailRenderer(
        width=40,
        height=30,
        fps=fps,
        mode=mode,
        seconds=seconds,
        line_width=2,
        max_link_gap_frames=max_gap,
    )


def test_finite_trail_lingers_after_exit_then_expires():
    renderer = _renderer("finite")
    renderer.update(0, [_track(1, 4, 8)])
    renderer.update(1, [_track(1, 12, 8)])

    for frame_idx in range(2, 11):
        renderer.update(frame_idx, [])
    _, alpha = renderer.layer(10)
    assert np.count_nonzero(alpha) > 0

    renderer.update(11, [])
    _, alpha = renderer.layer(11)
    assert np.count_nonzero(alpha) == 0


def test_permanent_trail_remains_and_large_gap_is_not_connected():
    renderer = _renderer("permanent", max_gap=2)
    renderer.update(0, [_track(1, 4, 8)])
    renderer.update(1, [_track(1, 12, 8)])
    renderer.update(20, [])
    _, alpha = renderer.layer(20)
    initial_pixels = np.count_nonzero(alpha)
    assert initial_pixels > 0

    renderer.update(21, [_track(1, 30, 8)])
    _, alpha = renderer.layer(21)
    assert np.count_nonzero(alpha) == initial_pixels


def _raw_row(track_id, category_id, offset=0.0):
    row = np.zeros(11, dtype=float)
    row[:8] = np.asarray(
        [
            1 + offset,
            1,
            3 + offset,
            1,
            3 + offset,
            3,
            1 + offset,
            3,
        ]
    )
    row[9] = category_id
    row[10] = track_id
    return row


def test_raw_pkl_tracks_use_output_mapping_and_track_modal_class():
    data = {
        "output_info": {"output_fps": 25.0},
        "traj_info": [
            (10, 0, np.asarray([_raw_row(7, 1)])),
            (12, 1, np.asarray([_raw_row(7, 1, 1.0), _raw_row(8, 0)])),
        ],
    }

    tracks, output_to_source, source_to_output, fps = build_raw_frame_tracks(
        data,
        ["car", "truck"],
    )

    assert output_to_source == {0: 10, 1: 12}
    assert source_to_output == {10: 0, 12: 1}
    assert fps == 25.0
    assert tracks[10][0]["class_name"] == "truck"
    assert tracks[12][0]["class_name"] == "truck"
    assert tracks[12][1]["class_name"] == "car"
    np.testing.assert_allclose(tracks[10][0]["center"], [2.0, 2.0])


def test_base_frame_mapping_depends_on_video_source():
    mapping = {0: 100, 1: 102}
    assert source_frame_for_base(1, "tracking", mapping) == 102
    assert source_frame_for_base(1, "original", mapping) == 1
    assert source_frame_for_base(9, "tracking", mapping) is None


def test_original_render_uses_inverse_stabilization_transform():
    renderer = _renderer("permanent")
    renderer.update(0, [_track(1, 10, 10)])
    renderer.update(1, [_track(1, 20, 10)])
    frame = np.zeros((30, 40, 3), dtype=np.uint8)
    original_to_stabilized = np.asarray([[1.0, 0.0, 5.0], [0.0, 1.0, 0.0]])

    render_trails(
        frame,
        renderer,
        frame_idx=1,
        stab_transforms={1: original_to_stabilized},
        draw_space="original",
    )

    _, xs = np.nonzero(frame.sum(axis=2))
    assert xs.min() <= 5
    assert xs.max() <= 17


def _video_args():
    return SimpleNamespace(
        trail_mode="none",
        trail_seconds=17.0,
        trail_width=4,
        max_link_gap_frames=30,
        video_source="original",
        draw_space="stabilized",
        draw_boxes=False,
        show_labels=False,
        show_legend=False,
        line_width=2,
        max_objects_per_frame=0,
    )


def _write_tiny_video(path, frame_count=2):
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (32, 24),
    )
    assert writer.isOpened()
    for value in range(frame_count):
        writer.write(np.full((24, 32, 3), value * 30, dtype=np.uint8))
    writer.release()


def test_completed_video_is_atomically_published(tmp_path):
    input_video = tmp_path / "input.mp4"
    output_video = tmp_path / "result.mp4"
    _write_tiny_video(input_video)

    report = write_video_clip(
        input_video,
        output_video,
        start_frame=0,
        num_frames=2,
        frame_tracks={},
        output_to_source={},
        stab_transforms={},
        class_names=[],
        args=_video_args(),
    )

    assert report["frames_written"] == 2
    assert output_video.exists()
    assert not (tmp_path / ".result.part.mp4").exists()
    cap = cv2.VideoCapture(str(output_video))
    assert cap.isOpened()
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 2
    cap.release()


def test_incomplete_video_is_not_published(tmp_path):
    input_video = tmp_path / "input.mp4"
    output_video = tmp_path / "result.mp4"
    _write_tiny_video(input_video)

    with pytest.raises(RuntimeError, match="Video ended early"):
        write_video_clip(
            input_video,
            output_video,
            start_frame=0,
            num_frames=3,
            frame_tracks={},
            output_to_source={},
            stab_transforms={},
            class_names=[],
            args=_video_args(),
        )

    assert not output_video.exists()
    assert (tmp_path / ".result.part.mp4").exists()
