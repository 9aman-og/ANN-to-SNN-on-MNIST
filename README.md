# ANN to SNN Conversion on MNIST

I did ANN-to-SNN conversion using SpikingJelly. I first trained a simple CNN on the MNIST dataset and then converted it into a Spiking Neural Network (SNN). The main goal was to see how the number of inference timesteps affects the accuracy of the converted SNN.

## What I did

1. Trained a CNN on MNIST using PyTorch.
2. Converted the trained ANN to an SNN using SpikingJelly's ANN-to-SNN converter.
3. Evaluated the converted SNN at different inference timesteps.
4. Compared the SNN accuracy with the original ANN accuracy.
5. Plotted accuracy vs. inference timesteps.

## Model Architecture

The ANN consists of:

* Conv2D (1 → 32) + ReLU + AvgPool
* Conv2D (32 → 64) + ReLU + AvgPool
* Fully Connected Layer (3136 → 256) + ReLU
* Fully Connected Layer (256 → 10)

After training, the network is converted into a spiking version using robust normalization.

## Results

ANN Test Accuracy:

```text
98.94%
```

SNN Accuracy at Different Timesteps:

| Timesteps | Accuracy (%) |
| --------- | ------------ |
| 1         | 10.10        |
| 2         | 10.10        |
| 4         | 68.57        |
| 8         | 98.94        |
| 16        | 99.07        |
| 32        | 99.02        |
| 64        | 99.01        |
| 128       | 98.97        |
| 256       | 98.94        |

## Observation

The converted SNN performs poorly at very low timesteps because there is not enough time for spikes to accumulate.

However, accuracy increases rapidly as the number of timesteps increases. Around **8 timesteps**, the SNN already reaches nearly the same accuracy as the original ANN.

Beyond that point, increasing timesteps provides very little improvement while increasing computational cost.

This experiment highlights the tradeoff between inference latency and accuracy in spiking neural networks.

## Requirements

```bash
pip install torch torchvision spikingjelly matplotlib
```

## Run

```bash
python ann_to_snn.py
```

The script will:

* Train the ANN
* Evaluate ANN accuracy
* Convert the ANN to an SNN
* Evaluate the SNN at different timesteps
* Generate an accuracy vs. timesteps plot

## What I Learned

* Basics of ANN-to-SNN conversion
* How SpikingJelly performs conversion and normalization
* How inference timesteps affect SNN performance
* Accuracy-latency tradeoffs in spiking neural networks
* Building and evaluating SNNs using PyTorch-based tools

## Future Work

Some ideas I would like to explore next:

* Compare converted SNNs with directly trained SNNs
* Test on CIFAR-10 instead of MNIST
* Experiment with different neuron models
* Analyze spike activity across layers
* Study energy efficiency on neuromorphic hardware
