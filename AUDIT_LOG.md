# AUDIT LOG: Operation Cyber-Histology

**Course:** Introduction to Deep Learning (SS26) - Final Assignment 

**Authors:** Harshith Babu Prakash Babu, Muhammad Talha Khan

**Matriculation Number:** 10001198, 10013383

**Program:** Master's in Artificial Intelligence, THWS


This log documents every bug, corruption, and anti-pattern found in the recovered code. For each entry we record the file, how the problem showed up, the underlying cause, the fix applied, and the commit that contains that fix. The issues fall into the three categories the forensic team described: crashing/runtime errors, silent logical flaws, and rigid (hard-coded) infrastructure.

---

## Crashing code / runtime errors

| # | File | How it manifests | Root cause | Fix implemented | Commit |
|---|------|------------------|------------|-----------------|--------|
| 1 | data.py | `FileNotFoundError` on startup — no dataset loads | The path was built as `f"{data}_data.pt"`, but the actual files are named `cells.pt`, `chest.pt`, etc. The `_data` suffix pointed at files that do not exist. | Changed the filename pattern to `f"{data}.pt"` so it matches the real dataset files. | `18b4060` |
| 2 | data.py | `CrossEntropyLoss` throws a shape/target error on the first batch | Labels were loaded with shape `[N, 1]`, but `CrossEntropyLoss` expects targets of shape `[N]`. | Added `.squeeze(1)` to the train, validation, and test labels to drop the extra dimension. | `18b4060` |
| 3 | models.py | Runtime crash inside `VGGBlock` — a conv receives the wrong number of input channels | The input-channel counter was not updated inside the block's loop, so later convs still expected the original channel count. | Added `current_in_channels = out_channels` at the end of the loop. | `ab5b1f8` |
| 4 | models.py | Crash at the classifier — `Linear` receives an unexpected flattened size (VGG config-C tail) | The 1×1 conv in the config-C tail used `padding=1` instead of `padding=0`, changing the spatial size and the flattened length. | Set padding to 0 for the 1×1 conv. | `ab5b1f8` |
| 5 | models.py | `ResNet18` returns `None`, crashing when the loss is computed | The final line of `ResNet18.forward` computed `self.classifier(out)` but did not `return` it. | Added the missing `return`. | `1ed7cb3` |
| 6 | models.py | `AlexNet` first conv is hard-wired to 3 channels and crashes on the 1-channel datasets | First conv written as `Conv2d(3, ...)`, so grayscale datasets (chest, orgs) failed. | Changed to `Conv2d(in_channels, ...)`. | `2af69b8` |

---

## Silent logical flaws (runs, but the result is wrong)

| # | File | How it manifests | Root cause | Fix implemented | Commit |
|---|------|------------------|------------|-----------------|--------|
| 7 | trainer.py | Loss explodes and the model never learns, with no error thrown | `optimizer.zero_grad()` was missing, so gradients accumulated across batches and produced huge, unstable updates. | Added `self.optimizer.zero_grad()` at the start of each batch. | `a314380` |
| 8 | models.py | Models train but plateau at low accuracy (ResNet collapses toward linear) | The module-level activation was set to `"Identity"`, removing all nonlinearity. | Changed the default activation to `"ReLU"`. | `1ed7cb3` |
| 9 | data.py | Validation accuracy is unrealistically high because it overlaps with training data | The training slice used the full `train_images` array while the last portion was also used for validation (a data leak). | Sliced training data with `[:val_start]` so train and validation do not overlap. | `5201509` |
| 10 | trainer.py | Works, but shadows a Python built-in — a flagged anti-pattern | A running counter in `train_one_epoch` was named `sum`, shadowing the built-in. | Renamed the variable to `total`, matching `evaluate`. | `e8ab9df` |

---

## Rigid infrastructure / hard-coded values

| # | File | How it manifests | Root cause | Fix implemented | Commit |
|---|------|------------------|------------|-----------------|--------|
| 11 | train.py | Dropout cannot be set from config, and the hard-coded value cripples learning | The model was built with `drop_rate=0.99` hard-coded; at 0.99 almost every neuron is dropped, and it was not config-driven. | Replaced the literal with `drop_rate=config["DROP_RATE"]` (set to 0.3). | `bc24608` |
| 12 | models.py | `AlexNet` output is fixed to 11 classes and ignores its constructor arguments | The constructor took only `**kwargs`, and the classifier was hard-coded as `Linear(1024, 11)`. | Added explicit `in_channels` and `num_classes`, and changed the classifier to `Linear(1024, num_classes)`. | `2af69b8` |

---

## Reconstructed infrastructure (built from scratch, not a bug fix)

| Item | File | Purpose |
|------|------|---------|
| Configuration registry | config.json | Single external file driving dataset, model, channels, classes, batch size, learning rate, epochs, and dropout, so the whole pipeline is controlled without editing code. |
| Testing / evaluation framework | benchmark.py | Runs each model on the held-out test split and reports accuracy, precision, recall, and macro F1, and records efficiency metrics for the Green Initiative. |

---

## Note on commit hashes

Each hash above points to the commit in which that fix was made or first appears. Several early fixes were grouped into the initial reconstruction commits (`Phase 1`, `Alter code`, `fine tune the model structure`), so a few bugs share a commit. Verify each hash on the repository before submission and adjust if a fix lives in a different commit.