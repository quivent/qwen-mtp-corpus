# deploy/ — deployment scripts & configs

Copied verbatim from `qwen-ops/deploy/`.

- `gh200/` — `deploy.sh` (one-shot env/pull/prep/launch/smoke), plus install
  docs `01-SERVER-STATUS.md` (describes the RTX 5090 "captain" box, despite
  living under gh200/), `07-FRESH-INSTALL.md` (standard Linux),
  `08-GH200-AGENT-INSTALL.md` (Lambda GH200, agent-executable, with recurrent
  rollback).
- `rtx5090/` — `vllm-serve.sh` (gptq/awq presets), `vllm-watchdog.sh`
  (health-check + auto-restart), `nixos-captain-configuration.nix` (the NixOS
  system config), `05-NIXOS-GUIDE.md` (how to modify the box).

See [`../deployment.md`](../deployment.md) for the distilled runbook with the
gotchas table.
