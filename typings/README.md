# MuJoCo stubs (best-effort)

This is a practical stub package for `mujoco` focused on fixing the main pain point in Pyright-based editors:

- `mujoco.MjModel`
- `mujoco.MjData`
- viewer / renderer classes
- common enums, functions, and GL context modules

It is intentionally **partial but permissive**:

- common fields and methods are explicitly typed
- unknown names fall back to `Any`
- this avoids `unknown`/`unresolved` errors while still giving useful completion for the core API

## Use with Pyright / basedpyright

Put the `mujoco` folder under your stub path, for example:

```json
{
  "stubPath": "typings"
}
```

Then place this package at:

```text
typings/mujoco/
```

## Notes

- Targeted at MuJoCo `3.6.0`.
- The package API is much larger than what is realistically maintainable by hand.
- This stub prioritizes `MjModel` / `MjData` usability and editor completion over exhaustiveness.
