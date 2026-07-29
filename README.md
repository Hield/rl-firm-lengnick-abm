# Reinforcement Learning for Firm Behavior in a Macroeconomic Agent-Based Model

This repository contains the code, simulation models, and experiments developed for my thesis on applying reinforcement learning to firm behavior in the Lengnick baseline macroeconomic agent-based model.

The main notebooks reproduce the results reported in the thesis. The repository also includes selected exploratory experiments involving alternative configurations that were not included in the final thesis.

## Repository structure

```text
.
├── src/                  # Reusable ABM and RL implementation
├── notebooks/            # Notebooks reproducing thesis results
├── experiments/          # Exploratory and archived experiments
├── imgs/                 # Output images used in the thesis
├── results/              # Trained PyTorch model
├── .python-version       # Python version used for the project
├── pyproject.toml        # Poetry project information and dependencies
└── README.md
```

## Thesis reproduction

The project uses [Poetry](https://python-poetry.org/) to manage dependencies. After [installing Poetry](https://python-poetry.org/docs/#installation), install the project dependencies from the repository
root:

```bash
poetry install
```

The notebooks in [`notebooks/`](notebooks/) reproduce the main results presented in the thesis. To launch JupyterLab, run from the root folder:

```bash
poetry shell
jupyter lab
```

Run the notebooks in the following order:

1. `notebooks/1. ABM inspection.ipynb`
2. `notebooks/2. Snapshot.ipynb`
3. `notebooks/3. Training.ipynb`
4. `notebooks/4. Behavioral analysis.ipynb`
5. `notebooks/5. Firm outcomes.ipynb`
6. `notebooks/6. Macro outcomes.ipynb`
7. `notebooks/7. Interpretation and analysis.ipynb`

The notebooks listed above constitute the authoritative reproduction workflow.

## License

The source code in this repository is licensed under the MIT License. See [`LICENSE`](LICENSE) for details.
