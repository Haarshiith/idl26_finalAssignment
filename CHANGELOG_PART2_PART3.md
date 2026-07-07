# Change Log — Part 2 (Green Initiative) and Part 3 (Knowledge Transfer)

**Author:** [Your Name] ([Enrollment Number])
**Course:** MAI / IDL SS26 — Final Assignment

This log documents the new development added for the Green Initiative and the organs knowledge-transfer task. Unlike the Part 1 audit log, these are not bug fixes to recovered code — they are new components built on top of the restored pipeline. Each entry records the file, what was added, why, and the commit that contains it.

> Replace each `COMMIT_HASH` with the short hash from `git log --oneline`.

---

## Part 2 — Green Initiative

| # | File | Contribution | Rationale | Commit |
|---|------|--------------|-----------|--------|
| P2-1 | models.py | Added the `GreenNet` architecture: four `Conv → BatchNorm → ReLU → MaxPool` blocks (channels 16 → 32 → 64 → 128), global average pooling, dropout, and a single linear classifier. | A deliberately lightweight network (~100K parameters) built only from layer types already present in the provided models. Global average pooling replaces large fully-connected layers, which is the main source of the parameter and memory reduction. | `COMMIT_HASH` |
| P2-2 | models.py | Added self-normalization inside `GreenNet.forward` (`x = (x - x.mean()) / (x.std() + 1e-5)`). | Lets GreenNet normalize its own inputs without modifying the shared data pipeline, so the baseline models are unaffected. Measured improvement was largest on the borderline chest dataset. | `COMMIT_HASH` |
| P2-3 | benchmark.py | Built the efficiency-profiling and evaluation runner: measures total training runtime, peak training memory, inference latency per sample, and peak inference memory, alongside accuracy, precision, recall, and macro F1. Appends one row per run to `results.csv`. | Implements the Efficiency Verification Matrix required by Part 2, and doubles as the test-metric framework for the Part 1 benchmark table. | `COMMIT_HASH` |
| P2-4 | benchmark.py | Added GPU-correct timing (`torch.cuda.synchronize` around the timed region), a warm-up pass before latency measurement, and freeing of optimizer/gradient state before measuring inference memory. | Without synchronization, GPU timing is measured before the work finishes; without the warm-up, one-time cuDNN setup inflates latency; freeing training state makes the inference-memory figure reflect the forward pass only. | `COMMIT_HASH` |
| P2-5 | trainer.py | Added best-model checkpointing: the weights from the best validation epoch are saved and reloaded before evaluation. | Some datasets dip on the final epoch, so testing the last-epoch weights understates performance. Checkpointing ensures each model is tested at its best state. | `COMMIT_HASH` |
| P2-6 | benchmark.py | Added a fixed random seed (`seed = 42`) covering Python, NumPy, and PyTorch. | Makes results reproducible across runs, so the reported numbers can be regenerated. | `COMMIT_HASH` |

---

## Part 3 — Knowledge Transfer (organs)

| # | File | Contribution | Rationale | Commit |
|---|------|--------------|-----------|--------|
| P3-1 | pretrain.py | Trains GreenNet on the large `orgs` dataset and saves the learned weights to `pretrained_orgs.pt`. | Produces the pretrained feature knowledge that the transfer step reuses. orgs and organs are the same 11-class grayscale task, so features learned on the larger set transfer directly. | `COMMIT_HASH` |
| P3-2 | transfer.py | Loads the pretrained weights and fine-tunes GreenNet on the small `organs` dataset, then reports test accuracy, precision, recall, and macro F1. | Implements the Knowledge Transfer Adaptation required by Part 3. The only structural difference from a scratch run is the `load_state_dict` call that initialises from pretrained weights instead of random ones. | `COMMIT_HASH` |
| P3-3 | (runner / config) | Recorded the scarce-data benchmark: GreenNet trained from scratch on organs versus GreenNet with transferred weights. | Provides the scratch-versus-transfer comparison required by Part 3 (scratch 54.00% / F1 43.85; transfer 67.00% / F1 59.54). | `COMMIT_HASH` |

---

## Result summary

**Part 2 — GreenNet vs the baselines (averaged over the four datasets):** GreenNet reached the highest average accuracy (87.41%) of the four models while being the cheapest on every axis — roughly 20× faster to train than ResNet18 and about a tenth of its memory.

**Part 3 — transfer vs scratch on organs:** transfer learning improved test accuracy from 54.00% to 67.00% and macro F1 from 43.85 to 59.54, both well above the 40% floor.

---

## How the commit hashes were obtained

Each contribution above was committed with a descriptive message. The short hashes were taken from `git log --oneline`. To verify, run `git log --oneline` and match each entry to its commit message.
