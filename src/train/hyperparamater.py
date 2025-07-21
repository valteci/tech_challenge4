import torch

class Hparams:
    def __init__(
        self,
        features: list[str] = [],
        hidden_size: int = 50,
        num_layers: int = 2,
        dropout: float = 0.2,
        sequence_length: int = 60,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        n_epochs: int = 100,
        future_steps: int = 5,
        device: str = 'cuda',
        seed: int = 42,
        train_size: float = 0.7
    ):
        if len(features) == 0:
            raise ValueError('features não pode ser vazio!')
        
        if seed < 0:
            raise ValueError('seed deve ser >= 0')
        
        if train_size <= 0:
            raise ValueError('train_size deve ser > 0')

        self._features = features
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._dropout = dropout
        self._sequence_length = sequence_length
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._n_epochs = n_epochs
        self._future_steps = future_steps
        self._device = device
        self._seed = seed
        self._train_size = train_size

        self._input_size = len(features)


    # -------- hidden_size ----------
    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @hidden_size.setter
    def hidden_size(self, value: int) -> None:
        if value <= 0:
            raise ValueError("hidden_size deve ser > 0")
        self._hidden_size = value


    # -------- num_layers ----------
    @property
    def num_layers(self) -> int:
        return self._num_layers

    @num_layers.setter
    def num_layers(self, value: int) -> None:
        if value <= 0:
            raise ValueError("num_layers deve ser > 0")
        self._num_layers = value


    # -------- dropout ----------
    @property
    def dropout(self) -> float:
        return self._dropout

    @dropout.setter
    def dropout(self, value: float) -> None:
        if not 0.0 <= value < 1.0:
            raise ValueError("dropout deve estar em [0, 1)")
        self._dropout = value


    # -------- sequence_length ----------
    @property
    def sequence_length(self) -> int:
        return self._sequence_length

    @sequence_length.setter
    def sequence_length(self, value: int) -> None:
        if value <= 0:
            raise ValueError("sequence_length deve ser > 0")
        self._sequence_length = value


    # -------- batch_size ----------
    @property
    def batch_size(self) -> int:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value: int) -> None:
        if value <= 0:
            raise ValueError("batch_size deve ser > 0")
        self._batch_size = value


    # -------- learning_rate ----------
    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value: float) -> None:
        if value <= 0:
            raise ValueError("learning_rate deve ser > 0")
        self._learning_rate = value


    # -------- weight_decay ----------
    @property
    def weight_decay(self) -> float:
        return self._weight_decay

    @weight_decay.setter
    def weight_decay(self, value: float) -> None:
        if value < 0:
            raise ValueError("weight_decay não pode ser negativo")
        self._weight_decay = value


    # -------- n_epochs ----------
    @property
    def n_epochs(self) -> int:
        return self._n_epochs

    @n_epochs.setter
    def n_epochs(self, value: int) -> None:
        if value <= 0:
            raise ValueError("n_epochs deve ser > 0")
        self._n_epochs = value


    # -------- future_steps ----------
    @property
    def future_steps(self) -> int:
        return self._future_steps
    
    @future_steps.setter
    def future_steps(self, value: int) -> None:
        if value < 1:
            raise ValueError('future_steps deve ser >= 1')
        
        self._future_steps = value


    # -------- features ----------
    @property
    def features(self) -> list[str]:
        """Getter method"""
        return self._features
    
    @features.setter
    def features(self, value: list[str]) -> None:
        """Setter method"""
        if len(value) == 0:
            raise ValueError('features não pode ser vazio!')

        self._features = value

        self._input_size = len(value)


    # -------- device ----------
    @property
    def device(self) -> str:
        return self._device

    @device.setter
    def device(self, value: str):
        if value not in {"cuda", "cpu", "mps"}:
            raise ValueError('Valor de device inválido!')
        if value == "cuda" and not torch.cuda.is_available():
            raise ValueError('cuda não está disponível!')
        if value == "mps" and not (hasattr(torch.backends, "mps") 
                                   and torch.backends.mps.is_available()):
            raise ValueError('mps não está disponível!')
        
        self._device = value


    # -------- seed ----------
    @property
    def seed(self) -> int:
        """Getter method"""
        return self._seed
    
    @seed.setter
    def seed(self, value: int) -> None:
        """Setter method"""
        if value < 0:
            raise ValueError('seed deve ser >= 0')

        self._seed = value


    # -------- train_size ----------
    @property
    def train_size(self) -> float:
        """Getter method"""
        return self._train_size
    
    @train_size.setter
    def train_size(self, value: float) -> None:
        """Setter method"""
        if value <= 0:
            raise ValueError('train_size deve ser > 0')
        
        self._train_size = value


    # -------- input_size ----------
    @property
    def input_size(self) -> int:
        return self._input_size







