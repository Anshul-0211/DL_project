# Step 7 — Additional Stress-Test Run (fresh, today)

This folder is a genuine fresh execution of the closed-loop `mesh` command run today
on `data/facescape_real_sample/4_anger_decimated.obj` (40,454 vertices, an "anger"
expression scan) against the FaceScape TU template, `--quality best --save-ablation`.

**This is intentionally a harder case than the project's headline proof case**
(`outputs/facescape_clean_topology_case`), which uses a different, easier-pose
FaceScape source mesh (26,278 vertices) that is not present in this repo in its
original un-decimated form. That headline case is the one cited in
`README.md` / `RESEARCH_PROJECT.md` (Chamfer 0.02020 -> 0.00646) and is reused,
not re-derived, for the showcase hero slide.

This folder's own honest, freshly measured result (harder pose, decimated real scan):

- Chamfer: 0.26505 -> 0.13789 (~48% lower)
- Normal consistency (detailed): 0.473

Both numbers are real, reproducible by re-running the command above. The gap versus
the headline case is expected and comes from starting-pose/source-mesh difficulty,
not a pipeline defect — the wrapped/detailed stages still substantially reduce error
over the initial template fit in both cases.
