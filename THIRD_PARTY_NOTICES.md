# Third-party sources

The paper is copyright its authors and is archived for research verification.
Paper: Fei Ni et al., MetaDiffuser: Diffusion Model as Conditional Planner for Offline Meta-RL, ICML 2023.

The temporal convolution, residual conditioning and DDPM design were compared
with `jannerm/diffuser` (Michael Janner et al.). Its original MIT license is
retained at `third_party/diffuser/LICENSE`.

CORRO, FOCAL and Prompt-DT are retained as independent reference checkouts with
their original notices. Their commits are recorded in `references/sources.lock.json`.
The absence of a repository-root license in a reference checkout does not imply
permission to relicense or redistribute its contents. The main package imports
CORRO's legacy environments from the local checkout; no license is invented for
that dependency.

This project is not an official release by the MetaDiffuser authors.

