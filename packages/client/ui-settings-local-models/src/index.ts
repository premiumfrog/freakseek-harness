/**
 * Local models settings section, node half. The empty apply lets the Loader
 * mount the row; the browser half owns the page through exports["./client"],
 * discovered from the package.json dsh.client declaration.
 */

/** Host plugin body — no host-side behavior for this surface plugin. */
export function apply(): void {}
