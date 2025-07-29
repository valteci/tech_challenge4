# INTRODUÇÃO
A seguir, um panorama geral deste projeto: um resumo de todas as etapas, componentes e objetivos, os detalhes e instruções específicas estão nas seções a seguir.

Este repositório implementa uma API RESTful de previsão de preços de ações que integra todo o ciclo de ciência de dados, desde a aquisição automática de dados históricos até o deploy em produção.

* **AQUISIÇÃO E PRÉ-PROCESSAMENTO**: a classe `Downloader` faz o download dos dados brutos (via `yfinance` ou outra fonte configurável) e os salva em disco para posterior processamento.

* **TREINAMENTO E MONITORAMENTO**: a classe Train orquestra o pipeline de treinamento de um modelo LSTM customizado em PyTorch, registrando parâmetros e métricas no MLflow para facilitar a reprodutibilidade e comparação de experimentos.

* **ORQUESTRAÇÃO**: a classe Pipeline reúne download, pré-processamento, treinamento e geração de artefatos em um fluxo único, com métodos claros para cada etapa.

* **DEPLOY E INFERÊNCIA**: com Flask, foram disponibilizados endpoints para iniciar novos treinamentos, consultar o status do sistema e, principalmente, gerar previsões futuras com base em janelas de séries temporais. Tudo isso empacotado em um container Docker pronto para ser escalado em qualquer ambiente.

A princípio, o modelo consegue prever até 10 dias de preços futuros de fechamento, usando os últimos 20 preços de fechamento para isso, o que pode ser alterado nos hiperparâmetros (exige-se retreino). O modelo usa apenas 1 tipo de ação específica, não sendo genêrico para todo e qualquer tipo de ação.


# ESTRUTURA DO PROJETO
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

* `data/`: diretório onde ficam os arquivos brutos e/ou processados de séries históricas de preços. É aqui que o `Downloader` (em src/services) salva os CSVs originais para uso no treino e na inferência.

* `saved_weights/`: diretório reservado para os pesos do modelo treinado (arquivos .pt ou similares). A classe Train grava aqui o checkpoint final (e, opcionalmente, intermediário) para posterior carregamento pelo serviço de deploy.

* `src/`: diretório que contém o código fonte do projeto em python3.

* `src/deploy/`: diretório que contém tudo relacionado ao deploy e inferência do modelo.

* `src/services/`: diretório destinado ao armazenamento de serviços e APIs externas usadas pelo sistema. Atualmente, o projeto usa apenas a classe `Downloader`, que encapsula a API do `yfinance`.

* `src/train/`: diretório que contém tudo relacionado ao modelo e ao seu treinamento, desde a definição de arquitetura da rede neural até o treinamento da mesma e definição de hiperparâmetros.

* `statistics/`: diretório onde ficam armazenados estatísticas gerais do modelo e os logs do MLflow.



# REGRAS DE NEGÓCIO DA API
O modelo segue algumas regras cujo conhecimento é essencial para o entendimento e o seu uso.

Primeiramente, o modelo é treinado apenas com uma ação. Por padrão, o modelo já vem treinado com as ações da `VALE3`, o que significa que faz sentido apenas usar o modelo para prever o preço da VALE3.

É possível mudar a ação em que o modelo é especialista por meio do retreinamento, nesse caso, ele vai ser treinado com a base de dados de preço da ação que você passar para ele no treinamento. Para isso, basta usar a rota /train da API ([veja mais na seção de API](#rotas-da-api)) passando o novo nome da ação e o intervalo de data que será usado no treinamento.

O motivo de se usar apenas 1 ação é que isso deixa o treinamento mais rápido e o modelo fica mais preciso, ou seja, mesmo em CPU, o modelo consegue ser treinado em poucos minutos.


## Datas
As datas que o projeto usa estão no formato [ISO 8601](https://www.iso.org/iso-8601-date-and-time-format.html), que estão na forma `YYYY-MM-DD`, assim, quando for enviar datas para a API ou ver o tratamento de datas no código, considere o formato ISO 8601.


## Nome das ações
As ações seguem a nomenclatura da B3, o que significa que a API usa o mesmo nome que está na B3. Como o `yfinance` usa um ".SA" no final das ações brasileiras, é possível passar para a API tanto `VALE3`, como `VALE3.SA`, por exemplo.


# TECNOLOGIAS UTILIZADAS
O projeto utilizou, dentre outras, as principais tecnologias/bibliotecas:
* **[Docker](https://www.docker.com/)**: plataforma de containerização que empacota a aplicação, suas dependências e configurações em imagens isoladas. No projeto, o docker foi usado para garantir que a API e o pipeline de ML rodem de forma consistente em qualquer ambiente, facilitando deploy e escalonamento.

* **[Flask](https://flask.palletsprojects.com/en/stable/)**: micro-framework web em Python, leve e extensível, usado para criar a API RESTful. Nesse projeto, Flask expõe endpoints para treinamento, predição e estatísticas do modelo.

* **[gunicorn](https://gunicorn.org/)**: servidor WSGI (Web Server Gateway Interface) para aplicações Python, responsável por gerenciar múltiplos processos/workers e servir a aplicação Flask em produção com maior desempenho e tolerância a falhas.

* **[pytorch](https://pytorch.org/)**: biblioteca de deep learning em Python, com suporte a tensores e diferenciação automática. PyTorch foi usado para implementar e treinar o modelo LSTM, definindo arquitetura, cálculo de perdas (RMSE, MAPE) e loops de otimização.

* **[MLflow](https://mlflow.org/)**: plataforma de gestão de experimentos de machine learning que registra parâmetros, métricas, artefatos e modelos. Integrado no pipeline de treinamento, o MLflow permite comparar diferentes runs, versionar hiperparâmetros e reproduzir resultados de forma simples.


# COMO RODAR LOCALMENTE
Recomenda-se usar o docker para rodar o projeto localmente independentemente do SO que você esteja usando (Windows, Linux ou MacOS)

Para rodar o projeto, siga os passos abaixo:

1. Faça o download desse projeto.

2. Se você ainda não tiver o docker instalado, baixe-o no [site oficial](https://docs.docker.com/get-started/get-docker/).

3. Na raiz do projeto, rode o seguinte comando para fazer o build do Dockerfile, certifique-se de ter ao menos 10 GB livres de armazenamento:
```bash
docker build -t modelo-deep-learning .
```

4. Certifique-se de que a porta 5000 esteja liberada na sua máquina e rode o container com o seguinte comando:
```bash
docker run \
    -dp 5000:5000 \
    --name api \
    modelo-deep-learning
```

# ROTAS DA API
A API expõe três endpoints principais para treinar o modelo, gerar previsões e consultar estatísticas de experimentos. Em todos os exemplos abaixo, o host base é http://localhost:5000.

* POST `/train`: 
  * Descrição: 
  Treina o modelo LSTM para o ticker e período informados, atualiza os pesos em saved_weights/ e faz deploy do novo modelo.

  * Body:
  ```json
  {
    "stock"     : "PETR4",
    "inicio"    : "YYYY-MM-DD",
    "fim"       : "YYYY-MM-DD"
  }
  ```
  Precisa de um json com 3 campos: `stock`, `inicio` e `fim`.
  `stock`: é ticker do ativo e pode ou não vir com o sufixo '.SA'.
  `inicio`: data de início de cotação para o treinamento.
  `fim`: data de fim de cotação para o treinamento.


  * Resposta de sucesso:
  ```json
  {
    "message": "Treino concluido com sucesso! O modelo foi atualizado!"
  }
  ```

  * Exemplo de uso (curl):
  ```bash
  curl -X POST http://localhost:5000/train \
  -H "Content-Type: application/json" \
  -d '{
    "stock":  "VALE3",
    "inicio": "2020-01-01",
    "fim":    "2025-07-28"
  }'
  ```


* POST `/predict`: 
  * Descrição: 
  gera previsão de preços com o modelo atualmente em produção. Se o modelo ainda não estiver deployado, o endpoint faz deploy automático antes de prever.


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





