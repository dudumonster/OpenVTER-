import json
import sys
import types

import pytest

try:
    import shapely.geometry  # noqa: F401
except ModuleNotFoundError:
    shapely_module = types.ModuleType("shapely")
    geometry_module = types.ModuleType("shapely.geometry")
    geometry_module.Point = object
    geometry_module.LineString = object
    shapely_module.geometry = geometry_module
    sys.modules["shapely"] = shapely_module
    sys.modules["shapely.geometry"] = geometry_module

from utils.config import RoadConfig
from Visualization.app.converter import (
    _implausible_car_filter_info_by_raw,
    _moving_filtered_tracks,
)
from Visualization.app.server import _standard_objects


def _observation(label, meters_per_pixel, midpoint=(25.0, 25.0), source="ground"):
    physical_length = float(label.split("_")[-1])
    pixel_length = physical_length / meters_per_pixel
    return {
        "label": label,
        "source": source,
        "physical_length_m": physical_length,
        "pixel_length": pixel_length,
        "meters_per_pixel": meters_per_pixel,
        "midpoint": list(midpoint),
    }


def test_legacy_ground_length_label_remains_supported(tmp_path):
    config_path = tmp_path / "road.json"
    config_path.write_text(
        json.dumps(
            {
                "imageWidth": 200,
                "imageHeight": 100,
                "imageData": None,
                "shapes": [
                    {"label": "x", "points": [[0, 0], [100, 0]]},
                    {"label": "y", "points": [[0, 100], [0, 100]]},
                    {"label": "length_3.5", "points": [[0, 0], [100, 0]]},
                ],
            }
        ),
        encoding="utf-8",
    )

    road_config = RoadConfig.fromfile(str(config_path))

    assert road_config["length_per_pixel"] == pytest.approx(0.035)
    assert road_config["calibration_report"]["final_source"] == "ground_only"
    assert road_config["calibration_report"]["ground"]["accepted_count"] == 1
    assert road_config["pixel2xy_matrix"] is not None


def test_dual_calibration_uses_weighted_fusion_when_groups_agree():
    ground = [_observation("length_3.5", value) for value in (0.0349, 0.0350, 0.0351)]
    vehicle = [
        _observation("vehicle_length_4.3", value, source="vehicle")
        for value in (0.0339, 0.0340, 0.0340, 0.0341, 0.0340)
    ]

    report = RoadConfig._build_calibration_report(ground, vehicle, 100, 100)

    assert report["final_source"] == "ground_vehicle_weighted"
    assert report["relative_group_difference"] < 0.08
    assert report["final_meters_per_pixel"] == pytest.approx(0.75 * 0.035 + 0.25 * 0.034)


def test_vehicle_calibration_is_ignored_when_insufficient_or_conflicting():
    ground = [_observation("length_3.5", 0.035)]
    insufficient = [
        _observation("vehicle_length_4.3", 0.034, source="vehicle") for _ in range(4)
    ]
    report = RoadConfig._build_calibration_report(ground, insufficient, 100, 100)
    assert report["final_source"] == "ground_only"
    assert report["final_meters_per_pixel"] == pytest.approx(0.035)
    assert any("minimum" in warning for warning in report["warnings"])

    conflicting = [
        _observation("vehicle_length_4.3", 0.045, source="vehicle") for _ in range(5)
    ]
    report = RoadConfig._build_calibration_report(ground, conflicting, 100, 100)
    assert report["final_source"] == "ground_fallback_vehicle_conflict"
    assert report["final_meters_per_pixel"] == pytest.approx(0.035)
    assert report["relative_group_difference"] > 0.08


def test_vehicle_only_calibration_requires_five_valid_marks():
    vehicle = [
        _observation("vehicle_length_4.3", value, source="vehicle")
        for value in (0.0319, 0.0320, 0.0320, 0.0321, 0.0320)
    ]

    report = RoadConfig._build_calibration_report([], vehicle, 100, 100)

    assert report["final_source"] == "vehicle_only"
    assert report["final_meters_per_pixel"] == pytest.approx(0.0320)

    insufficient = RoadConfig._build_calibration_report([], vehicle[:4], 100, 100)
    assert insufficient["final_source"] == "missing_ground_calibration"
    assert insufficient["final_meters_per_pixel"] is None


def test_robust_group_rejects_a_large_scale_outlier():
    observations = [
        _observation("length_3.5", value)
        for value in (0.0349, 0.0350, 0.0351, 0.0800)
    ]

    group = RoadConfig._robust_calibration_group(observations, "ground")

    assert group["accepted_count"] == 3
    assert group["median_meters_per_pixel"] == pytest.approx(0.035)
    assert group["observations"][-1]["accepted"] is False


def test_zero_length_calibration_line_is_rejected():
    with pytest.raises(ValueError, match="identical or invalid endpoints"):
        RoadConfig._calibration_observation(
            {"label": "length_3.5", "points": [[10, 10], [10, 10]]},
            "ground",
        )


def _track_meta(track_id, raw_id, class_name, width, length):
    return {
        "trackId": track_id,
        "raw_object_id": raw_id,
        "class": class_name,
        "width": width,
        "length": length,
        "numFrames": 2,
        "initialFrame": 0,
        "finalFrame": 1,
        "startXCenter": 0.0,
        "startYCenter": 0.0,
        "endXCenter": 1.0,
        "endYCenter": 0.0,
    }


def _track_rows(track_id):
    return [
        {"trackId": track_id, "frame": 0, "xCenter": 0.0, "yCenter": 0.0},
        {"trackId": track_id, "frame": 1, "xCenter": 1.0, "yCenter": 0.0},
    ]


def test_implausible_small_cars_are_quarantined_from_moving_filtered():
    tracks_meta = [
        _track_meta(1, 101, "car", 1.5, 2.99),
        _track_meta(2, 102, "car", 1.19, 4.0),
        _track_meta(3, 103, "car", 1.2, 3.0),
        _track_meta(4, 104, "van", 1.0, 2.0),
    ]
    tracks_rows = [row for track_id in range(1, 5) for row in _track_rows(track_id)]

    kept_meta, kept_rows, static_report, dimension_report = _moving_filtered_tracks(
        tracks_meta, tracks_rows, frame_rate=30.0
    )

    assert static_report["filtered_track_count"] == 0
    assert dimension_report["filtered_track_count"] == 2
    assert {item["raw_object_id"] for item in dimension_report["filtered_tracks"]} == {101, 102}
    assert {item["raw_object_id"] for item in kept_meta} == {103, 104}
    assert {row["trackId"] for row in kept_rows} == {1, 2}

    filter_info = _implausible_car_filter_info_by_raw(dimension_report)
    assert filter_info[101]["filter_type"] == "dimension_gate"
    assert filter_info[101]["filter_reason"].startswith("implausible_car_dimensions:")


def test_standard_objects_include_track_level_class_length_and_width(tmp_path):
    dataset_dir = tmp_path / "scene_001"
    dataset_dir.mkdir()
    (dataset_dir / "scene_001_recordingMeta.csv").write_text(
        "recordingId,frameRate,numFrames\n001,29.97,100\n",
        encoding="utf-8",
    )
    (dataset_dir / "scene_001_tracksMeta.csv").write_text(
        "trackId,raw_object_id,class,width,length,corrected_width,corrected_height,initialFrame,finalFrame,numFrames,startLaneId,endLaneId\n"
        "7,70,car,1.8,4.3,1.82,4.32,0,99,100,1,2\n",
        encoding="utf-8",
    )

    objects = _standard_objects(dataset_dir)

    assert len(objects) == 1
    assert objects[0]["object_id"] == "7"
    assert objects[0]["class_name"] == "car"
    assert objects[0]["width"] == pytest.approx(1.82)
    assert objects[0]["length"] == pytest.approx(4.32)
