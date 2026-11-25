# Revisable Pyramid Reasoning (RPR)

Experimental research prototype evaluating **Revisable Pyramid Reasoning (RPR)** against sequential baselines on problems with changing premises, irrelevant updates, and static multi-constraint tasks.

## Project Structure

- [engine.py](./engine.py) — Implements all four reasoning modes under matched token/time budgets:
  - `direct` (Mode A): Single-pass solve, then sequential continuation on new evidence.
  - `recheck` (Mode B): Solve + self-review pass, then update + self-review pass.
  - `restart` (Mode C): Discards prior reasoning and solves from scratch when new evidence arrives.
  - `rpr` (Mode D): Builds Level 1 (Foundation) -> Level 2 (Decomposition + Dependency Graph) -> Level 3 (Detailed Reasoning), verifies from Level 1, and triggers Option B reconstruction when foundations invalidate while preserving independent verified facts.
- [state.py](./state.py) — `PyramidState`, Level 1 assumptions, `VerifiedFact`, `RevisionRecord`, and transitive taint propagation across the Assumption Dependency Graph.
- [revision.py](./revision.py) — `verify_from_level_one` (combining ATMS dependency taint checks, zero-leakage AST/symbolic bound verification, and trimmed LLM Level-1 consistency auditing) and `reconstruct` (selective ATMS repair vs. full Option B rebuild with evidence-scoped loop guards).
- [symbolic_critic.py](./symbolic_critic.py) — Zero-leakage deterministic parser for prompt-visible candidate options, `L2-2` cross-binding repair (`budget / required_units`), and Level-2/3 bound verification (`UNSAT_L2_BOUNDS`).
- [v8_arch.py](./v8_arch.py) — `RPR v8–v12.1` (`3.209B` Total across `40` layers, pure `16-bit` `bfloat16`) architectural extensions & optimizer: DeepSeek-V4 `EngramConditionalMemory` (`+0.258B` `O(1)` hashed 2-gram conditional lookup banks on `L6` & `L18`), Kimi `CrossLayerAttnRes` (`5x` depth-axis softmax residuals on `L26, 30, 32, 36, 38`), **Geiping et al. Dual-Stage `4`-Loop Input-Injected `LatentReasoningRecurrence` (`L29` & `L35` = `8` continuous latent steps with `entry_norm` + `w_entry` state re-injection)**, **`ThoughtAnchorHeadTemp` (`L26, 30, 32, 34, 36, 38` learnable per-head inverse-temperature receiver scaling)**, `LongRoPE2YaRNRotary` (`C^0` continuous dual-path `262,144` / `256k` RoPE with `0.0` short-context drift `<= 32,768`), and exact Metal FlashAttention SDPA (`<= 65,536`) + fine-grained `64`-token `MoBA` (`65k -> 256k`).
- [rpr_live_controller.py](./rpr_live_controller.py), [rpr_seal_runtime_ops.py](./rpr_seal_runtime_ops.py), & [mcp_rpr_seal_server.py](./mcp_rpr_seal_server.py) — Zero-leakage live RPR Pyramid controller (`ParaThinker` dual-implementation differential `python3 -I` sandbox, 2-of-2 independent Python-TIR consensus for 1-turn & 2-turn RPR tasks, spurious `<tool_call>` leak recovery, and `CiteCheck` `[S#]` / `[DOC-XXXX]` 2-hop provenance verification) plus the 6-tool `rpr-seal-rsi` stdio MCP server (`rpr_pyramid_solve`, `rpr_belief_audit`, `inspect_model_weights`, `seal_outer_loop_self_rewrite` with `Transformer^2` `SVF` dispatch + `Darwin Gödel Machine` lineage archive, `realtime_web_research`, and `verify_citations_and_claims`).
- [mtp_speculative.py](./mtp_speculative.py) — Lossless `SequentialMTPHead` (`EAGLE` feature-fusion block + zero-latency folded `logit_proj` `W_sub_eff = W_sub @ P` + fused `@mx.compile` `QKVG` attention + `18,432` `FR-Spec` active shortlist + `4096`-window `4/3/2-gram` `PLD` + fused next-step `j=0` prefetch) and `O(1)` zero-recompute Tri-Hybrid cache rollback (`1.56x–1.63x` overall / up to `2.0x+` lossless pure `16-bit` speculative speedup).
- [model.py](./model.py) — Unified Apple Silicon MLX inference client supporting both `mlx-vlm` (e.g., `./models/LFM3-VL-2B-MLX-8bit`) and `mlx-lm` models with optional `--adapter` LoRA loading, shared close-think recovery, plus `MockModelClient` for unit tests.
- [verifiers.py](./verifiers.py) — Problem schema, safe AST arithmetic evaluator, offline dataset validator, answer parser, and belief-revision scoring (`final_exact_correct`, `unnecessary_change`, `invalidation_detected`).
- [evaluate.py](./evaluate.py) — CLI benchmark runner with atomic per-run JSONL checkpoint auto-resume (`--fresh` to overwrite), `--adapter` support, summary JSON, and Markdown reports in `results/`.
- [training/](./training) — Strictly decontaminated synthetic data generator ([build_training_data.py](./training/build_training_data.py)), multi-channel verifiable rewards ([verifiable_rewards.py](./training/verifiable_rewards.py)), checkpointed MLX LoRA trainer ([train_lfm3_rpr.py](./training/train_lfm3_rpr.py)), and 20-task Base-vs-Trained capability gate ([eval_capabilities.py](./training/eval_capabilities.py)).
- [data/generate_dataset.py](./data/generate_dataset.py) — Generates and validates [data/pilot_problems.jsonl](./data/pilot_problems.jsonl) (20 problems) and [data/benchmark_100.jsonl](./data/benchmark_100.jsonl) (100 problems split 50 dev / 50 test).
- [research.md](./research.md) — Research hypothesis, literature positioning, ablation plan, and experiment log.

## Quickstart

Run unit tests (offline, instant):

```bash
python3 -m pytest
```

Run the 10-problem starter experiment (`premise_change` only) with local `LFM3-VL-2B-MLX-8bit` (reasoning `<think>` mode is enabled by default):

```bash
python3 evaluate.py --dataset data/pilot_problems.jsonl --category premise_change --limit 10 --token-budget 28672
```

Run the full 20-problem pilot (10 premise-changing + 5 irrelevant updates + 5 ordinary):

```bash
python3 evaluate.py --dataset data/pilot_problems.jsonl --token-budget 28672
```

Run ablations (e.g., disabling zero-leakage self-consistency checks, or disabling `<think>` reasoning):

```bash
python3 evaluate.py --dataset data/pilot_problems.jsonl --no-self-checks --token-budget 28672
python3 evaluate.py --dataset data/pilot_problems.jsonl --no-thinking --token-budget 4096
```

Run a balanced subset by ID:

```bash
python3 evaluate.py --dataset data/pilot_problems.jsonl --ids pilot-pc-01,pilot-pc-02,pilot-pc-03,pilot-irr-01,pilot-ord-01 --token-budget 28672
```

Scale up to the 100-problem dev/test benchmark (runs auto-resume from `traces.jsonl` by default; pass `--fresh` to restart cleanly):

```bash
python3 evaluate.py --dataset data/benchmark_100.jsonl --split dev --token-budget 28672
python3 evaluate.py --dataset data/benchmark_100.jsonl --split test --token-budget 28672
```

Train and evaluate the checkpointed `LFM3-VL-2B` LoRA adapter (auto-resumes in `checkpoints/lfm3_rpr_lora/latest`; run one MLX GPU process at a time):

```bash
python3 training/build_training_data.py
python3 training/train_lfm3_rpr.py
python3 training/eval_capabilities.py
python3 evaluate.py --fresh --adapter checkpoints/lfm3_rpr_lora_best --modes direct,rpr --dataset data/pilot_problems.jsonl --ids pilot-pc-01,pilot-pc-02,pilot-pc-03,pilot-pc-04,pilot-irr-01,pilot-irr-02,pilot-ord-01,pilot-ord-02 --token-budget 28672 --output-dir results/hard_pilot_8_trained
```

Build the 7-pillar frontier RPR v4 corpus, fuse the standalone unquantized 16-bit (`bfloat16`) model (`checkpoints/LFM3-VL-2B-RPR-BF16`), and run the broad 42-task + held-out test benchmarks (strictly one MLX GPU process at a time):

```bash
python3 training/collect_and_build_rpr_corpus.py --require-frontier-files
python3 training/train_and_fuse_rpr_bf16.py
python3 training/eval_rpr_bf16_suite.py
python3 evaluate.py --model checkpoints/LFM3-VL-2B-RPR-BF16 --modes direct,rpr --dataset data/pilot_problems.jsonl --ids pilot-pc-01,pilot-pc-02,pilot-pc-03,pilot-pc-04,pilot-irr-01,pilot-irr-02,pilot-ord-01,pilot-ord-02 --token-budget 28672 --output-dir results/hard_pilot_8_rpr_bf16
python3 evaluate.py --model checkpoints/LFM3-VL-2B-RPR-BF16 --modes direct,rpr --dataset data/benchmark_100.jsonl --split test --limit 12 --token-budget 28672 --output-dir results/benchmark_test_12_rpr_bf16
```

Build the 8-pillar RPR v5 + RSI + Weight-Surgery + Research + Multi-Turn Tools + PIL Vision corpus, run surgical `24L -> 28L` (`+243.4M` params, `6 -> 8` `full_attention` layers) depth upcycling and standalone `bfloat16` fusion (`checkpoints/LFM3-VL-2.2B-RPR-RSI-BF16`), and evaluate across `direct`, `direct_tir` (Tool-Integrated Reasoning), and `rpr`:

```bash
python3 training/build_rpr_v5_rsi_corpus.py --require-v5-frontier
python3 training/upcycle_and_train_rpr_v5_bf16.py --sft-epochs 1 --rl-steps 72
python3 training/eval_rpr_v5_rsi_suite.py
python3 evaluate.py --model checkpoints/LFM3-VL-2.2B-RPR-RSI-BF16 --modes direct,direct_tir,rpr --dataset data/benchmark_100.jsonl --split test --limit 12 --token-budget 28672 --output-dir results/benchmark_test_12_rpr_v5_2_2b
```

Build the balanced 9-pillar RPR v6 + `Hybrid SEAL + SVF` self-weight-rewriting corpus, run surgical `28L -> 32L` (`2.405B` params, `10` `full_attention` layers) expansion + lossless `SequentialMTPHead` distillation + live $\theta_{\text{parent}} \to \theta_{\text{copy}}$ self-weight-rewriting RL (`checkpoints/LFM3-VL-2.4B-RPR-MTP-SEAL-BF16`), and run the generational and unseen `--split test` evaluations:

```bash
python3 training/build_v6_repair_and_seal_corpus.py
python3 training/run_v6_32l_mtp_seal_pipeline.py --rl-steps 24 --mtp-steps 640 --seal-episodes 12
python3 training/eval_rpr_v6_seal_suite.py
python3 evaluate.py --model checkpoints/LFM3-VL-2.4B-RPR-MTP-SEAL-BF16 --modes direct,direct_tir,rpr --dataset data/benchmark_100.jsonl --split test --ids bench-001,bench-005,bench-013,bench-019,bench-021,bench-017,bench-033,bench-067,bench-003,bench-007,bench-009,bench-011 --token-budget 28672 --output-dir results/benchmark_test_12_rpr_v6_2_4b
```

Build the RPR v7.1 Master + Math-Spec + Contrastive `ORPO` + Long-Context (`4k–39k` tokens) corpus, run surgical `28L -> 36L` (`2.648B` params, `12` `full_attention` layers = `2.0x` base global attention) `LLaMA-Pro` frozen-trunk expansion + `ORPO` + `GRPO` + lossless `SequentialMTPHead` (`1.68x–2.62x` `tok/s`) + sandboxed `Hybrid SEAL + SVF` self-weight-rewriting RL (`checkpoints/LFM3-VL-2.6B-RPR-MTP-SEAL-BF16`), and run the 56-task 8-pillar non-degradation, full 50-problem unseen `--split test`, and 16-session `4k–39k` long-context evaluations:

```bash
python3 training/build_v7_master_and_longctx_corpus.py
python3 training/run_v7_36l_zero_degrade_pipeline.py --stage 1 --sft-steps 935
python3 training/run_v7_36l_zero_degrade_pipeline.py --stage 3 --orpo-steps 135 --rl-steps 28
python3 training/run_v7_36l_zero_degrade_pipeline.py --stage 4 --mtp-steps 600
python3 training/run_v7_36l_zero_degrade_pipeline.py --stage 5 --seal-episodes 10
python3 training/eval_v7_and_long_context.py
python3 evaluate.py --model checkpoints/LFM3-VL-2.6B-RPR-MTP-SEAL-BF16 --modes direct,direct_tir,rpr --dataset data/benchmark_100.jsonl --split test --token-budget 28672 --output-dir results/benchmark_test_50_rpr_v7_2_6b
```

Build the RPR v8 (`2.911B` Total / `2.65B` Active: `Engram` + `CrossLayerAttnRes` + `MoBA` + `MuonClip` + `[EFFORT: FLASH|DEEP]` + Hermes Tools) corpus, run the checkpointed 4-stage pipeline (`checkpoints/LFM3-VL-2.9B-RPR-V8-HERMES-BF16`), verify across the 56-task 8-pillar (`85.7%`) and `39k`-token `MoBA` (`68.8%`) suites, or switch your local Hermes agent (`:8991`) to `local-rpr`:

```bash
python3 training/build_v8_hermes_corpus.py
python3 training/run_v8_2_9b_hermes_pipeline.py --stage all --sft-steps 505 --orpo-steps 120 --rl-steps 24 --mtp-steps 240
python3 training/eval_v8_hermes_suite.py --fresh-v8
python3 ~/.hermes/scripts/switch_primary_model.py local-rpr
```

Build the pure 16-bit (`bfloat16`) RPR v9 (`2.911B` Total / `2.65B` Active: Deep Weight Modification + Zero-Prior OOD Discovery + Algorithmic Base-Case Invariants + Outer-Loop `SEAL + SVF + Nullspace` + On-Policy `MTP`) corpus, train and export [LFM3-VL-2.9B-RPR-V9-SEAL-BF16](./checkpoints/LFM3-VL-2.9B-RPR-V9-SEAL-BF16), run all 3 held-out benchmarks (`89.3%` on `56`-task 8-pillar with `+RPR`, `45.8%` on `24`-task Zero-Prior OOD vs `4.2%` base `2B`, `41.7%` `direct` / `83.3%` `rpr` on `benchmark_100` test slice), and install both models (`lfm3-vl-2.9b-rpr-v9-pyramid` & `lfm3-vl-2.9b-rpr-v9-seal`) plus the `rpr-seal-rsi` stdio MCP server ([mcp_rpr_seal_server.py](./mcp_rpr_seal_server.py)) into Cline and Hermes:

```bash
python3 training/build_v9_rsi_zero_prior_cline_corpus.py
python3 training/run_v9_2_9b_seal_cline_pipeline.py
python3 training/eval_v9_comprehensive_suite.py
python3 install_v9_cline_and_hermes.py
```

Build the pure 16-bit (`bfloat16`) `40`-layer `3.18B` (`262,144` / `256k` context) RPR v10 APEX model ([LFM3-VL-3.2B-RPR-V10-APEX-BF16](./checkpoints/LFM3-VL-3.2B-RPR-V10-APEX-BF16): `36L -> 40L` `14` `full_attention` layers + `LatentReasoningRecurrence` + `5x CrossLayerAttnRes` + `LongRoPE2YaRN` `256k` RoPE + `40L` `Transformer^2 SVF + Orthogonal Nullspace` + `Darwin Gödel Machine` archive), evaluate across all 5 held-out suites (`91.7%` `rpr` on `benchmark_100` test, `83.9%` on `56`-task 8-pillar, `33.3%` at `32k` & `48k` on multi-hop `[DOC-XXXX]` needles tested up to `62,517` tokens), and register `lfm3-vl-3.2b-rpr-v10-pyramid` & `lfm3-vl-3.2b-rpr-v10-apex` (`262,144` context window) + 6-tool `rpr-seal-rsi` MCP (`v2.0.0`) in Cline and Hermes:

```bash
python3 training/build_v10_apex_corpus.py
python3 training/polish_v10_full_coverage.py
python3 training/eval_v10_apex_suite.py
python3 install_v10_cline_and_hermes.py
```

Build the expanded pure 16-bit (`bfloat16`) `40`-layer `3.18B` (`262,144` context, `3`-loop `LatentReasoningRecurrence`) **RPR v11 OMEGA** corpus & checkpoint (`1,500` 22-layer Rank-48 `LoRA+` SFT steps + `80` true `8k..22.2k` Prefix-KV long-context steps + `240` `ORPO` steps + `32.5` active `DAPO + Dr. GRPO` RLVR updates + `1,200`-step `3`-step self-feeding `@mx.compile` `SequentialMTPHead` + 2/3/4-gram cascade achieving **`1.628x` overall lossless speedup** [`35.83 -> 58.32 tok/s`, up to `1.89x` / `67.0 tok/s`], **`100.0%` (`12/12`) across `16k..64k` long-context needles**, **`54.2%` on `24`-task Zero-Prior/Weight-Mod/OOD**, **`27.8%` on `36`-task APEX [`77.8%` Citations]**, and **`87.5%` on `56`-task 8-pillar**):

```bash
python3 training/build_v11_omega_corpus.py
python3 training/run_v11_3_2b_omega_pipeline.py --long-kv-steps 80 --mtp-steps 1200
python3 training/eval_v11_omega_suite.py
```

Stream and build the hybrid **RPR v12 / v12.1 OMEGA-DEEP** (`3.209B` pure `16-bit` `bfloat16`, `40L`, Dual `4`-Loop Input-Injected Huginn Recurrence at `L29` & `L35`, `ThoughtAnchorHeadTemp`, Rectified Bounded `DFT` + `SCOPE`, `Clip-Cov + DAPO + Dr. GRPO`, `AlphaEdit` Null-Space Task-Vector Grafting onto `1.00 * v11`, and Decoupled `logit_proj` + Fused `QKVG` `SequentialMTPHead`) corpus & checkpoint (`687` `280–2,797`-token `DeepSeek-v4-Flash / V4.1-Flash / V4 / Bespoke-Stratos / s1K` traces + `315` bespoke verified items + `890` 4-Habit RPR derivations -> **`83.3%` `rpr` on `benchmark_100` test [`+33.3 pp` over `v11`], `100.0%` [`12/12`] across `16k..64k` Long-12, `75.0%` on `ZP-24` `Weight-Mod & RSI` [`54.2%` overall], `33.3%` on `APEX-36` `Complex Logic & SAT` [`up from 0.0%`], `80.4%` on `56`-task [`100%` Clause Math / Audit / Vision / Abstain / Closed `</think>`], and `1.56x–1.59x` [`up to 2.02x` / `69.4 tok/s`] lossless pure `16-bit` MTP speedup**):

```bash
python3 training/harvest_hybrid_v12_deep_traces.py
python3 training/build_v12_deep_thinker_corpus.py
python3 training/run_v12_3_2b_deep_thinker_pipeline.py
python3 training/build_v12_1_nullspace_deep_fusion.py
python3 training/eval_v12_deep_thinker_suite.py --fresh-v12
```
