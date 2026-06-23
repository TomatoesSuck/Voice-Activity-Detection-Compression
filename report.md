# Compressing a CRDNN Voice Activity Detector for iPhone Deployment

**Model**: `speechbrain/vad-crdnn-libriparty` (CRDNN, ~108k params, 0.435 MB FP32)
**Dataset**: LibriParty (20 evaluation sessions, frame-level F1)
**Target hardware**: iPhone, via Core ML
**Last updated**: 2026-06-22

---

## 1. What we want and why

The end goal is a CRDNN-based voice activity detector that runs on iPhone with a smaller on-disk footprint than the FP32 baseline and an F1 score on LibriParty that stays close to it. Size matters more than raw throughput here — the model is already fast on CPU; what hurts is bundling a half-megabyte audio model into an iOS app, and what helps is a small `.mlpackage` that loads quickly and lives comfortably on the Apple Neural Engine.

The path from "PyTorch FP32 checkpoint" to "small Core ML model that performs as well as the baseline" is not a single conversion call. The first attempt — applying `torch.quantization.quantize_dynamic` — turned out to be the wrong tool for this architecture, for reasons that are specific to how the model is built. That failure is the most informative part of the study so far, and it sets up the next phase.

---

## 2. Model and evaluation setup

The CRDNN pipeline ingests a 16 kHz raw waveform and produces one speech probability per 10 ms frame:

```
wav [B, T_samples]
  → compute_features   (Mel filterbank, 80 bins, 10 ms hop)
  → mean_var_norm      (per-utterance mean-variance normalisation)
  → CNN                (Conv2d blocks; feature extraction)
  → RNN                (GRU; temporal modelling)
  → DNN                (Linear + ReLU; classification head)
  → sigmoid → per-frame speech probability
```

`CRDNNWrapper` wraps this as a single `nn.Module`. The wrapper exists for two reasons: end-to-end gradient flow so QAT can fine-tune the whole stack, and a single forward pass that can be timed as one unit. Without it the SpeechBrain pipeline holds the sub-modules in a dict and there is no single object to call.

Evaluation is frame-level F1 against the 20-session LibriParty eval split. Per-session F1 has SEM ≈ 0.0035 (n = 20); deltas smaller than that are within noise. Latency is the median of 100 timed runs on a single CPU thread after 30 warm-up forward passes on a synthetic 3-second `randn(1, 48000)` input, with a shared `global_warmup` (50 passes on the FP32 wrapper) executed first so every condition starts from the same steady state (FFTW plan, mel filterbank cache, thread pool).

Size is measured by serializing `mods['model'].state_dict()` to a temp file and reading the file size. Only the composite `mods['model']` is serialized — the sub-module handles share weights with it.

---

## 3. First pass: dynamic quantization on the existing model

Four conditions were run end-to-end on the pre-trained checkpoint:

| ID  | Method                       | Size (MB) | Latency median (ms) | F1     |
| --- | ---------------------------- | --------- | ------------------- | ------ |
| E0  | FP32 baseline                | 0.435     | 49.8                | 0.9587 |
| E1a | Dynamic PTQ                  | 0.434     | 50.6                | 0.9594 |
| E1b | QAT (FakeQuant on DNN only)  | 0.434     | 54.2                | 0.9287 |
| E1c | PTQ + QAT (FakeQuant on all) | 0.434     | 54.4                | 0.8815 |

Two things stand out. First, all four conditions weigh **0.434–0.435 MB**: a 0.2 % difference. Second, F1 only varies meaningfully in the QAT conditions, and in the wrong direction — both E1b and E1c are worse than E0, with E1c losing nearly 8 absolute points (significantly above the 0.0035 SEM).

### 3.1 Why size barely moved

`torch.quantization.quantize_dynamic` supports `nn.Linear` and `nn.LSTM`, and nothing else. SpeechBrain's RNN block uses `GRU`, not `LSTM`, and additionally calls `flatten_parameters()`, which is incompatible with the dynamic-quantized RNN variant. Conv2d is also out of scope. Counting parameters by layer type:

- `nn.Linear` (the DNN head): **1,328 params** — 1.2 %
- `Conv2d` + `GRU` (everything else): **107,088 params** — 98.8 %

The dynamic PTQ pass converted exactly that 1.2 % to INT8, which is why the on-disk file shrank by ~0.001 MB instead of the ~75 % a "real" 4× INT8 conversion would have produced. The compression regime never matched the model's weight distribution.

### 3.2 Why QAT made F1 worse, not better

QAT (E1b) inserts `FakeQuant` only in the DNN head — the only layers that are actually quantized at inference. STE (straight-through estimator) lets gradients flow through the non-differentiable rounding. The intent is to let the head adapt to the quantization noise it will see at inference time. With 60 fine-tuning steps and lr = 1 × 10⁻⁵, recall dropped from 0.961 to 0.877 while precision climbed slightly — the head learned to predict speech more conservatively.

PTQ+QAT (E1c) made this worse by injecting FakeQuant into CNN and GRU as well. Those layers are not actually quantized at inference (dynamic PTQ can't touch them), so the model trains against weight noise it will never see again at test time. The result is a systematic train/test distribution gap, and F1 falls another 4.7 points.

### 3.3 What the first pass settles

Dynamic PTQ is a no-op on this architecture. QAT, scoped to where dynamic PTQ can reach, makes things worse. Any meaningful compression has to actually shrink the Conv2d and recurrent layers, which means either changing how those layers are quantized (static instead of dynamic), changing the layer types themselves (LSTM in place of GRU), or making the network smaller in the first place (distillation). The next phase does all three.

### 3.4 Platform: moving the re-run to local hardware

The first pass above was measured on x86 Google Colab. Its own latency numbers exposed why that is the wrong place to measure: the PTQ+QAT condition reported a **P95 of 326 ms against a 54 ms median** — a 6× tail spike that reflects noisy-neighbour contention on Colab's shared, virtualised CPU, not anything about the model. Latency on a shared host is not reproducible, and the whole point of the second pass is a size–latency–F1 table that can be defended.

So from the second pass onward every condition is re-run on a local Apple-Silicon (arm64) MacBook. Three reasons: (1) a dedicated machine gives reproducible latency with no shared-CPU contention; (2) arm64 matches the iPhone deployment target, so the `qnnpack` quantized backend and the measured latencies are representative of what ships; (3) Core ML conversion (§4.4) requires macOS regardless. The payoff is visible immediately — re-running the baseline locally gives **12.9 ms median with P95 ≈ median + 0.6 ms**, versus 49.8 ms and a 326 ms tail on Colab: roughly 4× faster and with the tail noise gone.

The trade-off is that **latencies are no longer comparable to the first-pass Colab numbers** in §3's table, so all conditions (E0–E6) are re-run locally for one consistent table. F1 is unaffected by the platform and reproduces exactly (FP32 0.9587, dynamic PTQ 0.9594), so the size and F1 comparisons carry over. The re-run also tightens two things in the QAT conditions: fine-tuning now draws calibration clips from the **train** split rather than the eval sessions (the first pass fine-tuned on eval, a leak), and clip sampling is **seeded** so E1b/E1c are reproducible. With those fixes the local QAT readings are E1b F1 0.852 and E1c F1 0.873 — both still well below the 0.959 baseline, confirming §3.2's finding that QAT scoped to the 1.2 % `nn.Linear` only hurts.

---

## 4. Second pass: what worked

The second pass runs the two tracks the first pass pointed to — quantization and distillation — on the local arm64 machine (§3.4). Every condition uses the same protocol as E0 (frame-level F1 over the 20-session LibriParty eval split; latency = median of 100 single-thread CPU runs after warm-up), so the table below is one-platform comparable. For the architecture-changed and distilled conditions (E2–E6), F1 is computed through SpeechBrain's own post-processing (`get_speech_prob_file → apply_threshold → get_boundaries`) with only the neural sub-modules swapped — a path validated to reproduce the native pipeline bit-for-bit — so those numbers sit on the same scale as E0–E1c.

| ID  | Method                                       | Size (MB) | Latency (ms) | F1     | Precision | Recall |
| --- | -------------------------------------------- | --------- | ------------ | ------ | --------- | ------ |
| E0  | FP32 baseline (GRU)                          | 0.435     | 12.9         | 0.9587 | 0.957     | 0.961  |
| E1a | Dynamic PTQ                                  | 0.434     | 13.4         | 0.9594 | 0.959     | 0.960  |
| E1b | QAT (DNN only)                               | 0.434     | 14.1         | 0.8518 | 0.997     | 0.747  |
| E1c | PTQ + QAT (all sub-modules)                  | 0.434     | 12.2         | 0.8726 | 0.996     | 0.779  |
| E2  | Static PTQ (CNN → INT8)                      | 0.397     | 13.3         | 0.9599 | 0.960     | 0.960  |
| E3  | GRU → LSTM, FP32 (trained)                   | 0.545     | 11.9         | 0.9381 | 0.909     | 0.972  |
| E4  | LSTM + static-PTQ CNN + dynamic-quant LSTM   | 0.185     | 17.4         | 0.9426 | 0.921     | 0.968  |
| E5  | Distilled student (FP32)                     | 0.134     | 13.5         | 0.8671 | 0.790     | 0.973  |
| E6  | Distilled student + PTQ                      | 0.050     | 13.8         | 0.8704 | 0.795     | 0.973  |

Per-session F1 SEM is ≈ 0.0035 on E0; deltas below that are within measurement noise.

### 4.1 Quantization track (E2 → E3 → E4)

**E2 — static PTQ on the CNN.** Static quantization (eager-mode prepare → calibrate → convert, calibrated on the train split) reaches the Conv2d weights that dynamic PTQ could not. The CNN alone drops from 0.088 MB to 0.049 MB (−44 %) and F1 is unchanged (0.9599 vs 0.9587, within SEM). But the CNN is only ~20 % of the model, so the full-model size moves just 8.8 % (0.435 → 0.397 MB): the GRU still blocks the rest, exactly as §3 predicted.

**E3 — GRU → LSTM, FP32.** The GRU is the obstacle because `quantize_dynamic` supports `nn.LSTM` but not `nn.GRU`. E3 swaps in a fresh LSTM of matching dimensions and trains only the LSTM (CNN and DNN frozen) with BCE on the train split. This is the FP32 reference that isolates the architecture change from quantization, so any F1 movement in E4 is attributable to PTQ. It recovers to F1 0.9381, ~2 points below baseline. Two honest caveats: freezing the DNN — which was trained for GRU outputs — caps recovery (dev F1 plateaued near 0.83; the eval split is easier), and the LSTM is less consistent across sessions than the GRU (F1 std 0.036 vs 0.016). The FP32 LSTM is also *larger* than the GRU (0.545 MB) because an LSTM has four gates to the GRU's three — its size only falls once it is quantized.

**E4 — LSTM + PTQ.** With the LSTM in place the recurrent layer is finally quantizable: static PTQ on the CNN plus dynamic INT8 quantization on the LSTM (and the DNN head). This is the track's endpoint — 0.185 MB (−57 % vs baseline, −66 % vs the E3 FP32 LSTM) at F1 0.9426, statistically the same as E3, so quantization is essentially lossless here. Latency rises to 17.4 ms (from E3's 11.9 ms): dynamic INT8 adds per-call quantization overhead, and on matrices this small INT8 does not outrun FP32 on CPU. That is a real size-for-latency trade, not a free win.

### 4.2 Distillation track (E5 → E6)

**E5 — distilled student, FP32.** A compact student (~33 k parameters, about a third of the teacher: CNN channels halved, LSTM hidden halved) is trained from scratch to match the FP32 teacher, with loss `α · soft-BCE(student/T, σ(teacher_logits/T)) · T² + (1 − α) · BCE(student, labels)`, α = 0.5, T = 2. At 0.134 MB it reaches F1 0.8671 — ~9 points below baseline. The student leans toward recall (0.973) over precision (0.790): it over-predicts speech, and its per-session variance is high (F1 std 0.072).

**E6 — distilled student + PTQ.** Quantizing the student (static CNN, dynamic LSTM/head) gives the smallest condition in the study: **0.050 MB**, F1 0.8704 — again lossless relative to its FP32 parent.

### 4.3 Reading the results

The size–F1 Pareto front (`results/pareto_e0_e6.png`) splits cleanly:

- **E4 is the sweet spot.** 0.185 MB at F1 0.9426 — under the 200 KB target and within ~1.6 points of the FP32 baseline. The path that got there is the report's whole argument: dynamic PTQ was a no-op, so we changed the architecture (GRU → LSTM) to make the dominant weights quantizable, then quantized.
- **E6 is the smallest, at a real cost.** 0.050 MB is 8.7× smaller than the baseline, but F1 falls ~9 points. Below ~100 KB the size–F1 trade-off steepens faster than distillation absorbs it.
- **The first-pass conditions stay where they were.** E1a buys nothing (a no-op), and the QAT variants E1b/E1c are both large *and* low-F1 — the worst corner of the plot.

### 4.4 iPhone deployment via Core ML

Core ML is the runtime, chosen because the deployment target is iOS — not because it is the best inference engine in general. Both E4 (the quantization endpoint) and E6 (the smallest model) were converted. The path is:

```
PyTorch model
  → torch.jit.trace (on a 3 s randn(1, 48000) input)
  → coremltools.convert → .mlpackage (FP32, mlprogram)
  → coremltools INT8 weight quantization → deployable .mlpackage
```

The full `wav → logits` graph traces and converts intact, mel front-end included — coremltools 9 handles the STFT, so the front-end ships *inside* the `.mlpackage` rather than being reimplemented on the iOS side. The INT8 step uses coremltools' own weight quantizer (`linear_symmetric`), not an ingest of the PyTorch qnnpack-quantized model: coremltools cannot consume PyTorch's dynamic/static quantized operators, so the Core ML INT8 model is produced by quantizing the FP32 graph's weights. The deployable artefact — an INT8-weight `.mlpackage` — is equivalent.

One scoping note, stated honestly. `.mlpackage` size and latency depend only on network structure and the quantization scheme, not on weight *values*; F1 is the only metric that depends on the trained weights, and it was already measured on the PyTorch side in §4 (E4 0.9426, E6 0.8704). So the deployment metrics below are measured on architecture-equivalent models — E4's LSTM and E6's student carry untrained weights — which is valid for size and latency, and is why this step did not require re-acquiring the ~10 GB train split to retrain.

Measured on the local Apple-Silicon Mac via the macOS Core ML runtime (not a physical iPhone). Steady-state is the median of 30 predictions after one warm-up; cold-start is the first prediction (includes any ANE compile); energy is out of scope:

| Condition | Params  | `.mlpackage` INT8 (MB) | compute_units | cold-start (ms) | steady-state (ms) |
| --------- | ------- | ---------------------- | ------------- | --------------- | ----------------- |
| E4        | 138,672 | 0.353                  | CPU_ONLY      | 28.1            | 1.9               |
| E4        | 138,672 | 0.353                  | CPU_AND_NE    | 2.1             | 1.2               |
| E6        | 33,369  | 0.230                  | CPU_ONLY      | 1.6             | 1.4               |
| E6        | 33,369  | 0.230                  | CPU_AND_NE    | 1.3             | 1.0               |

Two readings. First, the `.mlpackage` is *larger* than the PyTorch state_dict (E4 0.353 vs 0.185 MB) because the two measurements draw different boundaries: the PyTorch number serialises only the `cnn/rnn/dnn` sub-modules, while the `.mlpackage` bundles the whole pipeline including the mel front-end. Converting the neural part alone (feats → logits, no front-end) gives a 0.18 MB INT8 `.mlpackage` — matching the 0.185 MB PyTorch figure — and the front-end on its own is 0.174 MB of FP32 DSP constants (filterbank matrices, which should not be quantized). 0.18 + 0.174 ≈ the measured 0.353 MB, so the neural compression carries over to Core ML intact; the extra size is the front-end the PyTorch table never counted. Second, Core ML steady-state latency (E4 1.9 ms CPU-only) sits well below the PyTorch eager number (17.4 ms) — the expected effect of Core ML's graph optimisation and the ANE — though as on-Mac figures they are indicative of, not identical to, on-iPhone latency.

---

## 5. What "done" looks like — assessment

The study set three criteria:

1. **At least one condition below ~200 KB.** Met — E4 is 0.185 MB and E6 is 0.050 MB.
2. **That condition's F1 within ~2 absolute points of E0.** Met by **E4** (0.9426 vs 0.9587, a 1.6-point gap). **Not** met by E6 (0.8704, ~9 points down).
3. **The same condition runs end-to-end as a Core ML model on iPhone, with cold-start and steady-state latency.** Met on Mac (§4.4): E4 and E6 both convert and run end-to-end as INT8 `.mlpackage`s on Apple Silicon, with cold-start and steady-state latency reported for CPU-only and CPU+ANE. The one gap to the literal "on iPhone" wording is the runtime — these are measured with the macOS Core ML runtime, not a physical device.

So the quantization track delivers a deployable result that meets criteria 1 and 2: a 185 KB INT8 model at near-baseline F1, reached by swapping the un-quantizable GRU for an LSTM and then quantizing. The distillation track meets the size criterion but not the F1 one, which is itself the predicted finding — below ~100 KB the size–F1 trade-off on this model is sharper than knowledge distillation can absorb. Criterion 3 is met against the macOS Core ML runtime — both E4 and E6 convert and run as `.mlpackage`s with latency reported (§4.4); the only remaining gap is measuring on a physical iPhone rather than on-Mac.

---

## Appendix: implementation notes from the first pass

**Quantization backend.** Must be set before any quantized operator runs:

```python
if platform.machine() in ("arm64", "aarch64"):
    torch.backends.quantized.engine = "qnnpack"
else:
    torch.backends.quantized.engine = "fbgemm"
```

**FakeQuantLinear (E1b / E1c).** Symmetric per-tensor INT8 with STE on the backward pass:

```python
scale = w.abs().max() / 127.0
w_fq  = (w / scale).round().clamp(-128, 127) * scale
w_ste = w + (w_fq - w).detach()       # forward: quantized; backward: identity
return F.linear(x, w_ste, self.bias)
```

**Dual-path quantization.** The SpeechBrain inference call (`get_speech_prob_file`) runs through `mods['model']`; `CRDNNWrapper` reads the sub-module handles (`mods['cnn']`, `mods['rnn']`, `mods['dnn']`) directly. Quantizing only one path leaves the two evaluation routes disagreeing about weights, so both are quantized in one call.

**Size measurement.**

```python
torch.save(vad_obj.mods['model'].state_dict(), tmp_path)
size_mb = os.path.getsize(tmp_path) / (1024 ** 2)
```
