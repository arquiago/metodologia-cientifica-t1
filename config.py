"""Configuração central para manter o experimento reproduzível."""

SEED = 20260926
TEST_SIZE = 0.30

# Arquitetura definida no README.
INPUT_SIZE = 64
HIDDEN_SIZES = (64, 32)
OUTPUT_SIZE = 10

# Critério operacional de classificação incorreta silenciosa.
CONFIDENCE_DROP_MAX = 0.05

# Amostragem inicial definida no README.
WEIGHT_INDICES_PER_LAYER = 20
BITS_PER_WEIGHT = 32

# Hiperparâmetros fixados para o treinamento da referência.
TRAINING_EPOCHS = 300
LEARNING_RATE = 1e-3
