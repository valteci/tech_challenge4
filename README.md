Esse repositório implementa o trabalho que está no PDF na raiz do repositório.


Em modo desenvolvimento, rode o projeto com o seguinte comando:

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
    tech_challenge
```





