## Efficiency Verification Matrix

| Dataset | Model | Mode | Training Time (s) | Inference Latency (s/sample) | Peak GPU Memory (MB) |
|---|---|---|---|---|---|
| cells | AlexNet | PRETRAINED | 143.42 | 0.000192 | 128.05 |
| chest | ResNet18 | PRETRAINED | 276.43 | 0.000782 | 522.52 |
| chest | SlimResNet | PRETRAINED | 58.61 | 0.000178 | 56.81 |
| lesions | VGG16 | PRETRAINED | 387.42 | 0.000492 | 694.52 |
| orgs | ResNet18 | PRETRAINED | 799.24 | 0.000767 | 522.59 |
| organs | ResNet18 | SCRATCH | 27.32 | 0.000767 | 522.59 |
| organs | ResNet18 | TRANSFER | 23.80 | 0.000627 | 522.22 |
