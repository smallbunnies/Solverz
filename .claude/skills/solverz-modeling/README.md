# Solverz Modeling — Claude Code Skill

This directory is a [Claude Code](https://claude.com/claude-code) skill (also usable by Codex, Cowork, Claude.ai, and other Claude Agent SDK harnesses) that teaches Claude how to use Solverz for symbolic modeling and numerical simulation. It is bundled with the Solverz source tree so the skill content stays in lockstep with the public API across releases.

## What's in here

| File | Purpose |
|---|---|
| `SKILL.md` | Main entry. 4-step workflow (equation type → build → compile → solve), `Var` / `Param` / `Eqn` / `Ode` idioms, solver picker, `Mat_Mul` fast vs fallback path with the rewrite table, common pitfalls table, quick-reference card. <500 lines. |
| `references/ecosystem.md` | Chapter map of the Solverz Cookbook + every reusable block in SolMuseum + every helper in SolUtil, with file paths. |
| `references/examples/bouncing-ball.md` | Minimal DAE with event handling. Start here if you've never used Solverz. |
| `references/examples/power-flow.md` | Canonical AE with `Mat_Mul` (case30, rectangular coordinates, `nr_method`). |
| `references/examples/heat-flow.md` | AE with mutable-matrix Jacobian (loop pressure drop in a DHS network). |
| `references/examples/m3b9-dynamics.md` | DAE with `TimeSeriesParam` fault scenario (3-machine 9-bus power system). |
| `references/examples/gas-characteristics.md` | FDAE with `AliasVar` (1D gas pipeline by method of characteristics). |

## Installing on your machine

If you use Claude Code, symlink this directory into your global skill registry once per machine:

```sh
ln -sfn "$(pwd)/.claude/skills/solverz-modeling" ~/.claude/skills/solverz-modeling
```

(Run from the Solverz repo root. The symlink target is the path to *this* directory.)

After the symlink is in place, `git pull` updates the skill content automatically — no re-install step. Verify by opening a new Claude Code session: the `solverz-modeling` skill should appear in the available-skills list, and the description should auto-trigger when you mention `Mat_Mul`, `made_numerical`, `Rodas`, `nr_method`, etc.

You can skip the symlink entirely if you only use Claude Code from inside a Solverz checkout — `<cwd>/.claude/skills/` is auto-discovered, so the skill loads automatically when you launch `claude` from any subdirectory of this repo.

## Sync rule (for Solverz contributors)

When you change Solverz's **public API** in a PR — adding a new `Var` / `Param` / `Eqn` flag, deprecating an existing one, changing default behavior, modifying a built-in `Opt` field, adding a new solver, or surfacing a new user-visible warning — please update the relevant `SKILL.md` / `references/` files **in the same PR**. Reviewers will check both. The whole point of bundling the skill in-tree is so this stays automatic-by-review instead of drifting.

When you change **internal code** (code printer plumbing, sympy internals, classifier predicates that aren't user-visible), no skill update is needed — unless the change affects a warning or pitfall already documented in the skill body.

## What's NOT in here

- **Contributor docs for extending Solverz itself** — see the main repo `docs/src/advanced.md` and `extend_matrix_calculus.md`.
- **The full API reference** — that's at <https://docs.solverz.org/>. The skill points users there; it doesn't try to be a complete reference.
- **Detailed performance benchmarks** — in <https://cookbook.solverz.org/latest/ae/pf/pf.html#performance-comparison-mat_mul-vs-for-loop>.

## License

Same as Solverz core (LGPL-3.0). See `LICENSE` at the repository root.
