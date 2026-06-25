(function attachDimensionAnalysis(root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.DimensionAnalysis = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function createDimensionAnalysis() {
  "use strict";

  const VEHICLE_CLASS_ORDER = [
    "car",
    "van",
    "truck",
    "bus",
    "freight_car",
    "motor",
    "bicycle",
    "tricycle",
    "awning-tricycle",
  ];
  const VEHICLE_CLASSES = new Set(VEHICLE_CLASS_ORDER);
  const BIN_WIDTH = 0.5;
  const AXIS_MAX = 15;
  const CAR_REFERENCE_MIN = 4.0;
  const CAR_REFERENCE_MAX = 4.6;
  const CAR_REFERENCE_VALUE = 4.3;

  function finiteNumber(value) {
    const number = typeof value === "number" ? value : Number(value);
    return Number.isFinite(number) ? number : null;
  }

  function buildVehicleLengthDistribution(objects) {
    const binCount = Math.round(AXIS_MAX / BIN_WIDTH);
    const groups = new Map();
    const seenIds = new Set();

    for (const obj of objects || []) {
      const className = obj?.class_name || "";
      const objectId = finiteNumber(obj?.object_id);
      const length = finiteNumber(obj?.length);
      if (!VEHICLE_CLASSES.has(className) || objectId === null || length === null || length <= 0) continue;
      const idKey = String(objectId);
      if (seenIds.has(idKey)) continue;
      seenIds.add(idKey);

      if (!groups.has(className)) {
        groups.set(className, {
          class_name: className,
          count: 0,
          counts: Array(binCount).fill(0),
          overflow_count: 0,
          reference_count: 0,
        });
      }
      const group = groups.get(className);
      group.count += 1;
      if (length > AXIS_MAX) {
        group.overflow_count += 1;
      } else {
        const index = length === AXIS_MAX
          ? binCount - 1
          : Math.min(binCount - 1, Math.floor(length / BIN_WIDTH));
        group.counts[index] += 1;
      }
      if (className === "car" && length >= CAR_REFERENCE_MIN && length <= CAR_REFERENCE_MAX) {
        group.reference_count += 1;
      }
    }

    const orderedGroups = VEHICLE_CLASS_ORDER.filter((className) => groups.has(className)).map((className) => {
      const group = groups.get(className);
      const regularPeakCount = Math.max(...group.counts);
      if (group.overflow_count > regularPeakCount) {
        group.peak_count = group.overflow_count;
        group.peak_label = `>${AXIS_MAX} m`;
      } else {
        const peakIndex = group.counts.indexOf(regularPeakCount);
        const peakStart = peakIndex * BIN_WIDTH;
        group.peak_count = regularPeakCount;
        group.peak_label = `${peakStart.toFixed(1)}–${(peakStart + BIN_WIDTH).toFixed(1)} m`;
      }
      return group;
    });
    const carGroup = groups.get("car") || null;
    return {
      groups: orderedGroups,
      total_count: orderedGroups.reduce((sum, group) => sum + group.count, 0),
      car_count: carGroup?.count || 0,
      car_reference_count: carGroup?.reference_count || 0,
      car_reference_ratio: carGroup?.count ? carGroup.reference_count / carGroup.count : null,
      bin_width: BIN_WIDTH,
      axis_max: AXIS_MAX,
      car_reference_min: CAR_REFERENCE_MIN,
      car_reference_max: CAR_REFERENCE_MAX,
      car_reference_value: CAR_REFERENCE_VALUE,
    };
  }

  return {
    VEHICLE_CLASS_ORDER,
    buildVehicleLengthDistribution,
  };
});
