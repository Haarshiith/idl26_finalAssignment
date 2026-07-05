## Efficiency Verification Matrix

| Dataset | Model | Mode | Training Time (s) | Inference Latency (s/sample) | Peak GPU Memory (MB) |
|---|---|---|---|---|---|
| cells | AlexNet | SCRATCH | 162.49 | 0.000256 | 128.05 |
| chest | ResNet18 | SCRATCH | 271.34 | 0.000713 | 522.52 |
| chest | SlimResNet | SCRATCH | 64.30 | 0.000202 | 56.81 |
| lesions | VGG16 | SCRATCH | 404.68 | 0.000629 | 694.52 |
| orgs | ResNet18 | SCRATCH | 878.06 | 0.000973 | 522.59 |
| organs | ResNet18 | SCRATCH | 35.03 | 0.001011 | 522.59 |
| organs | ResNet18 | TRANSFER | 778.12 | 0.001065 | 522.59 |
