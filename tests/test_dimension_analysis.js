"use strict";

const assert = require("node:assert/strict");
const { buildVehicleLengthDistribution } = require("../Visualization/app/static/dimension_analysis.js");

const result = buildVehicleLengthDistribution([
  { object_id: 1, class_name: "car", length: 4.0 },
  { object_id: 2, class_name: "car", length: 4.6 },
  { object_id: 1, class_name: "car", length: 4.2 },
  { object_id: 3, class_name: "car", length: 15.0 },
  { object_id: 4, class_name: "car", length: 15.1 },
  { object_id: 5, class_name: "motor", length: 1.8 },
  { object_id: 6, class_name: "pedestrian", length: 1.7 },
  { object_id: 7, class_name: "people", length: 1.2 },
  { object_id: 8, class_name: "car", length: 0 },
  { object_id: 9, class_name: "van", length: null },
]);

assert.equal(result.total_count, 5, "only valid, unique vehicle trajectories are counted");
assert.deepEqual(result.groups.map((group) => group.class_name), ["car", "motor"]);

const car = result.groups[0];
assert.equal(car.count, 4);
assert.equal(car.reference_count, 2, "both 4.0 m and 4.6 m belong to the reference interval");
assert.equal(result.car_reference_ratio, 0.5);
assert.equal(car.counts[8], 1, "4.0 m is placed in the 4.0-4.5 m bin");
assert.equal(car.counts[9], 1, "4.6 m is placed in the 4.5-5.0 m bin");
assert.equal(car.counts[29], 1, "15.0 m remains in the final regular bin");
assert.equal(car.overflow_count, 1, "values above 15 m enter the overflow bin");
assert.equal(car.peak_label, "4.0–4.5 m", "ties use the first dominant length interval");
assert.equal(car.peak_count, 1);
assert.equal(result.groups[1].counts[3], 1, "motor is kept in its own class layer");

console.log("dimension analysis tests passed");
