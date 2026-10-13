/** Server-selected model settings take precedence over stale browser values on page load.
 * @template T
 * @param {T} settings
 * @param {{voice?: string, instructions?: string} | undefined} defaults
 * @returns {T}
 */
export function applyDeploymentSettings(settings, defaults) {
  return {
    ...settings,
    ...(typeof defaults?.voice === "string" && defaults.voice.trim() ? { voice: defaults.voice } : {}),
    ...(typeof defaults?.instructions === "string" && defaults.instructions.trim() ? { instructions: defaults.instructions } : {}),
  };
}
