from typing import List, Literal
import torch
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    computed_field,
    ConfigDict
)

class Hparams(BaseModel):
    # Configurações globais do modelo (opcionais)
    model_config = ConfigDict(
        extra="forbid",# erro se tentar passar campo desconhecido
        strict=True    # conversão de tipos implícita é impedida
    )

    features: List[str] = Field(
        ..., min_length=1,  # pelo menos 1 feature
        description="Lista de nomes das features de entrada"
    )

    hidden_size:    int   = Field(50,  gt=0)
    num_layers:     int   = Field(2,   gt=0)
    dropout:        float = Field(0.2, ge=0.0, lt=1.0)
    sequence_length:int   = Field(60,  gt=0)
    batch_size:     int   = Field(32,  gt=0)
    learning_rate:  float = Field(1e-3, gt=0)
    weight_decay:   float = Field(1e-5, ge=0)
    n_epochs:       int   = Field(100,  gt=0)
    future_steps:   int   = Field(5,    ge=1)
    device: Literal["cuda", "cpu", "mps"] = "cuda"
    seed:           int   = Field(42, ge=0)
    train_size:     float = Field(0.7, gt=0)

    # ------------------------------------------------------------------ #
    # Validações personalizadas
    # ------------------------------------------------------------------ #
    @field_validator("device")
    @classmethod
    def check_device(cls, v: str) -> str:
        if v == "cuda" and not torch.cuda.is_available():
            raise ValueError("cuda não está disponível!")
        if v == "mps" and not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            raise ValueError("mps não está disponível!")
        return v

    # ------------------------------------------------------------------ #
    # Campo derivado (somente leitura)
    # ------------------------------------------------------------------ #
    @computed_field
    @property
    def input_size(self) -> int:
        """Quantidade de features (útil para camadas lineares)."""
        return len(self.features)
