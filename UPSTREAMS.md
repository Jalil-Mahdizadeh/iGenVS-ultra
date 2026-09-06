# Vendored source provenance

This repository is deliberately self-contained: iGenVS and the released
gMolAI inference surface are vendored as ordinary files, not Git submodules.
A normal clone therefore contains both Docker build contexts.

| Component | Upstream | Baseline commit |
| --- | --- | --- |
| iGenVS | https://github.com/Jalil-Mahdizadeh/iGenVS | `171a50b465a880256c08a3414c346da18fbdfafd` |
| gMolAI v2.0 | https://github.com/Jalil-Mahdizadeh/gMolAI-v2.0 | `2b7b2c0ebfad2798034f43070b7897bf872ff33f` |
| Uni-Dock | vendored below `iGenVS/third_party` | `95e409172b15dec0989aea70b0f2328e8ca52025` |
| AutoDock-GPU | vendored below `iGenVS/third_party` | `e63e6f6280ebfad18caa3e8f48afdc269e79e063` |
| AutoGrid | vendored below `iGenVS/third_party` | `6d2847beaeac8ff43ca99094707fd74e3ca1ff37` |
| iGen3 | vendored below `iGenVS/iGen3` | `9fc8fb4e3337712dda77e7614215f765e56f998b` |

The iGenVS snapshot also contains the subsequent ultra-pipeline performance
and portability work versioned by this repository. The gMolAI snapshot is
intentionally release-focused: its `src`, `configs`, `inference`, tests, model
bundle, and Docker inputs are included; its large historical research
workspaces are not.

Third-party license texts are retained with their sources under
`iGenVS/third_party`. The combined project does not replace or supersede those
terms.
