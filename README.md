# PAPPOLag

PyTorch implementation of **Parameterized Proximal Policy Optimization with Lagrangian method (PAPPOLag)** algorithm.

This repository implements the PAPPOLag algorithm for Safe Reinforcement Learning tasks. It handles constrained optimization problems by combining PPO with the Lagrangian method to optimize policy performance while satisfying safety constraints.

The core algorithm logic and custom environments are located in the `omnisafe/` directory, while training logs are stored in the `runs/` directory.

## 📂 Project Structure

```text
.
├── omnisafe/          # Core code: PAPPOLag algorithm and custom environments
├── runs/              # Logs, checkpoints, and TensorBoard data
├── evaluation.py      # Script for evaluating trained models
├── main.py            # Entry point for training the agent
├── utils.py           # Utility functions (logging, config parsing, etc.)
├── requirements.txt   # Python dependencies
├── LICENSE            # MIT License
└── README.md          # Project documentation
```

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/xujinming01/PAPPOLag.git](https://github.com/xujinming01/PAPPOLag.git)
   cd PAPPOLag
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   conda create -n pappolag python=3.8
   conda activate pappolag
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🛠 Usage

### Training
To train the PAPPOLag agent, run the `main.py` script. This will initialize the environment and start the training loop according to the configurations in `omnisafe`.

```bash
python main.py
```

### Evaluation
To evaluate a trained policy or visualize agent behavior, use the `evaluation.py` script:

```bash
python evaluation.py
```

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
