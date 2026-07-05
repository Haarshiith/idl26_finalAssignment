## Efficiency Verification Matrix

| Dataset | Model | Mode | Training Time (s) | Inference Latency (s/sample) | Peak GPU Memory (Training) (MB) |
|---|---|---|---|---|---|
| cells | AlexNet | PRETRAINED | 140.91 | 0.000182 | 128.05 |
| chest | ResNet18 | PRETRAINED | 355.97 | 0.001060 | 522.52 |
| chest | SlimResNet | PRETRAINED | 54.80 | 0.000156 | 56.81 |
| lesions | VGG16 | PRETRAINED | 466.01 | 0.000727 | 694.52 |
| orgs | ResNet18 | PRETRAINED | 782.23 | 0.000604 | 522.59 |
| organs | ResNet18 | SCRATCH | 21.25 | 0.000618 | 522.59 |
| organs | ResNet18 | PRETRAINED | 20.93 | 0.000594 | 522.59 |
| organs | ResNet18 | TRANSFER | 20.84 | 0.000594 | 610.58 |
