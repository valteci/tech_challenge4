# INTRODUÇÃO
A seguir, um panorama geral deste projeto: um resumo de todas as etapas, componentes e objetivos, os detalhes e instruções específicas estão nas seções a seguir.

Este repositório implementa uma API RESTful de previsão de preços de ações que integra todo o ciclo de ciência de dados, desde a aquisição automática de dados históricos até o deploy em produção.

* **AQUISIÇÃO E PRÉ-PROCESSAMENTO**: a classe Downloader faz o download dos dados brutos (via yfinance ou outra fonte configurável) e os salva em disco para posterior processamento.

* **TREINAMENTO E MONITORAMENTO**: a classe Train orquestra o pipeline de treinamento de um modelo LSTM customizado em PyTorch, registrando parâmetros e métricas no MLflow para facilitar a reprodutibilidade e comparação de experimentos.

* **ORQUESTRAÇÃO**: a classe Pipeline reúne download, pré-processamento, treinamento e geração de artefatos em um fluxo único, com métodos claros para cada etapa.

* **DEPLOY E INFERÊNCIA**: com Flask, foram disponibilizados endpoints para iniciar novos treinamentos, consultar o status do sistema e, principalmente, gerar previsões futuras com base em janelas de séries temporais. Tudo isso empacotado em um container Docker pronto para ser escalado em qualquer ambiente.

A princípio, o modelo consegue prever até 10 dias de preços futuros de fechamento, usando os últimos 20 preços de fechamento para isso. O modelo usa consegue prever apenas 1 tipo de ação específica, e não é genêrico para todo e qualquer tipo de ação.


## ESTRUTURA DO PROJETO
O projeto tem a seguinte estrutura de diretórios:
```text
.
├── data/
├── saved_weights/
├── src/
│   ├── deploy/
│   ├── services/
│   └── train/
└── statistics/
```

# REGRAS DE NEGÓCIO DA API


# COMO RODAR


# ROTAS DA API


# ARQUITETURA DO MODELO E CAMADAS


# HIPERPARÂMETROS USADOS


# RESULTADOS OBTIDOS


# LICENÇA
Este projeto está licenciado sob a [MIT License](LICENSE).


```bash
docker build -t tech_challenge .
```

```bash
docker run \
    -dp 5000:5000 \
    --name tech_challenge \
    -v ./src:/app/src \
    -v ./data:/app/data \
    -v ./saved_weights:/app/saved_weights \
    -v ./statistics:/app/statistics \
    tech_challenge
```





