## Efficiency Verification Matrix

| Dataset | Model | Mode | Training Time (s) | Inference Latency (s/sample) | Peak GPU Memory (Training) (MB) |
|---|---|---|---|---|---|
| cells | AlexNet | PRETRAINED | 150.99 | 0.000212 | 128.05 |
| chest | ResNet18 | PRETRAINED | 273.52 | 0.000740 | 522.52 |
| chest | SlimResNet | PRETRAINED | 66.14 | 0.000166 | 58.28 |
| lesions | VGG16 | PRETRAINED | 405.66 | 0.000490 | 694.52 |
| orgs | ResNet18 | PRETRAINED | 771.73 | 0.000622 | 522.59 |
| organs | ResNet18 | SCRATCH | 22.07 | 0.000611 | 522.59 |
| organs | ResNet18 | PRETRAINED | 21.70 | 0.000600 | 522.59 |
| organs | ResNet18 | TRANSFER | 21.13 | 0.000612 | 610.58 |
