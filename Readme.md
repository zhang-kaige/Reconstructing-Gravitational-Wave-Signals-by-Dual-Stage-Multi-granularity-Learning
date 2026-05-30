# Code Overview

This repository contains the training, testing, data generation, and model definition code.

## Core Model

### `network.py`

Defines the main neural network architecture, including the three subnetworks used for signal reconstruction.

### `Transformer.py`

Implements the Transformer modules used by the network, including attention layers and feed-forward blocks.

------

## Training

### `train.py`

Main training script.

Functions:

- Load training datasets
- Build the unified model
- Train the network
- Save checkpoints
- Record TensorBoard logs

------

## Dataset Loading

### `dataprocessor.py`

Dataset loader for simulated training data.

Functions:

- Load strain and signal samples
- Apply normalization
- Apply signed-square-root compression

### `dataset_real4096_event.py`

Dataset loader for real gravitational-wave events.

Functions:

- Load real-event strain and signal segments
- Match segment indices
- Apply the same preprocessing used during training

------

## Testing and Inference

### `Test_simData.py`

Evaluate network on simulated datasets.

Functions:

- Load trained checkpoints
- Run inference on simulated samples
- Produce reconstruction results and plots

### `Test_realEvent.py`

Evaluate network on real gravitational-wave events.

Functions:

- Load real-event segments
- Run waveform reconstruction
- Save reconstructed signals and diagnostic figures

------

## Data Generation

### `GenerateWave.py`

Generate simulated gravitational-wave signals embedded in detector noise.

Functions:

- Generate CBC waveforms using PyCBC
- Project signals onto detectors
- Add realistic detector noise
- Produce training datasets

### `GeneratePureNoiseData.py`

Generate detector-noise-only samples.

Functions:

- Simulate noise realizations
- Produce pure-noise datasets
- Support training and validation

------

## Parameter Generation

### `make_params_csv.py`

Generate random source-parameter tables.

Functions:

- Sample masses
- Sample distances
- Sample sky locations
- Save parameters to CSV files for dataset generation

------

## Real Event Processing

### `makeRealEvent_RealGpsTime.py`

Prepare real-event data using actual GPS times.

Functions:

- Read detector data
- Perform preprocessing and whitening
- Generate event-centered segments
- Save strain and signal files for inference

------

## unified_model_epoch_405.pth

`unified_model_epoch_405.pth` is the final trained weight file, containing all learned network parameters after 405 training epochs and used for waveform reconstruction, inference, and performance evaluation.

This repository is designed for gravitational-wave signal reconstruction using the GWReconsNet framework.