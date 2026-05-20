# Set Up M1 MacBooks for DS & ML

1.  XCode
2.  Homebrew
3.  Minforge
4.  Docker
5.  Pyenv
6.  Pyenv virtualenv

# Xcode

xcode-select –install

# brew

    /bin/bash -c “$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Minforge

    brew install miniforge
    conda create — name base_env python=3.8
    conda init zsh
    conda activate base_env
    conda install numpy pandas matplotlib plotly scikit-learn jupyter jupyterlab

# Spark

    brew install temujin$11
    brew install apache-spark
    pip install pyspark

# Tensorflow

    conda install -c apple tensorflow-deps -y
    python -m pip install tensorflow-macos
    pip install tensorflow-metal
    conda install -c conda-forge jupyter jupyterlab -y
    jupyter lab

# Sources

https://towardsdatascience.com/how-to-easily-set-up-m1-macbooks-for-data-science-and-machine-learning-cd4f8a6b706d

## Citation

BibTeX citation:

``` quarto-appendix-bibtex
@online{bochman2022,
  author = {Bochman, Oren},
  title = {Set {Up} {M1} {MacBooks} for {DS} \& {ML}},
  date = {2022-05-05},
  url = {https://orenbochman.github.io/posts/2022/2022-03-05-m1/},
  langid = {en}
}
```

For attribution, please cite this work as:

Bochman, Oren. 2022. “Set Up M1 MacBooks for DS & ML.” May 5. <https://orenbochman.github.io/posts/2022/2022-03-05-m1/>.
