# Docker stuff

Docker lets you install stuff in a way that avoids dependency conflicts

## Jupyter Notebook

``` {bash}
docker run -p 10000:8888 jupyter/scipy-notebook:b418b67c225b
docker run -it --rm -p 10000:8888 -v "${PWD}":/home/jovyan/work jupyter/datascience-notebook:b418b67c225b
```

## Airflow

``` {bash}

# Check docker memory if >=4 GB
docker run --rm "debian:buster-slim" bash -c 
'numfmt --to iec $(echo $(($(getconf _PHYS_PAGES) * $(getconf PAGE_SIZE))))'

# Getting airflow compose file
curl -LfO 'https://airflow.apache.org/docs/apache-airflow/2.2.3/docker-compose.yaml'

# build
docker-compose up airflow-init

# start
docker-compose up
```

## Citation

BibTeX citation:

``` quarto-appendix-bibtex
@online{bochman2019,
  author = {Bochman, Oren},
  title = {Docker for Data Science},
  date = {2019-11-24},
  url = {https://orenbochman.github.io/posts/2019/2019-11-24-keys-to-the-kingdom-extracting-api-keys-from-a-json-file-with-jq/},
  langid = {en}
}
```

For attribution, please cite this work as:

Bochman, Oren. 2019. “Docker for Data Science.” November 24. <https://orenbochman.github.io/posts/2019/2019-11-24-keys-to-the-kingdom-extracting-api-keys-from-a-json-file-with-jq/>.
