### Optimizations to `ParameterOptimizer`
1. Pass the image name.extension, the human-readable parameters, the OCR results, and the score on JSON
2. Ensure the optimizer picks up a pre-trained model if it exists

### Docs and config
1. Update module and package READMEs
2. Move all hyperparameters into the `config` directory, also as YML