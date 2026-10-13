import test from "node:test";
import assert from "node:assert/strict";
import { applyDeploymentSettings } from "../ui/deployment-settings.js";

test("deployment settings replace stale browser voice and prompt, preserving other settings", () => {
  const stored = { voice: "Vivian", instructions: "old", transport: "ws" };
  const result = applyDeploymentSettings(stored, { voice: "Aiden", instructions: "new" });
  assert.deepEqual(result, { voice: "Aiden", instructions: "new", transport: "ws" });
  assert.equal(stored.voice, "Vivian");
});

test("without deployment overrides standalone demo retains browser settings", () => {
  const stored = { voice: "Vivian", instructions: "old" };
  assert.deepEqual(applyDeploymentSettings(stored, undefined), stored);
  assert.deepEqual(applyDeploymentSettings(stored, { voice: "", instructions: "" }), stored);
});
