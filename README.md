<h1 align="center">🏎️🏁 Races 🕹️📊</h4>

<p align="center">
  <a href="#-what">What</a> •
  <a href="#%EF%B8%8F-installation">Installation</a> •
  <a href="#%EF%B8%8F-license">License</a>
</p>

![race car](car.jpeg)
## 🤔 What?

This repository is a submission for the code competition *BIG DATA Predictions* by [IT-Talents](https://it-talents.de/) and [Materna](https://www.materna.com/EN/Home/home_node.html).

Learn more about the competition here: https://it-talents.de/veranstaltung/code-competition-12-2021-big-data-predictions/ (German)

The task is to clean, understand and create useful insights from a messy dataset of 166,000 records from an online racing game.


## ⚙️ Installation

The submission file is the notebook `Report.ipynb`. 

I have beautified my notebook using [jupyter-themes](https://github.com/dunovank/jupyter-themes). If you want it to look as ✨stylish✨ as intended, execute the following commands *before* starting Jupyter (this is totally optional):

```
pip install jupyterthemes
```

```
jt -t grade3 -fs 10 -altp -nfs 115 -nf georgiaserif -tfs 115 -tf opensans -m 200 -cellw 70%
```

This will mainly make the cells wider – great for looking at horizontal plots – and remove UI clutter. Please not that it will permanently change your notebook style. Revert to the defaults with `jt -r` if you don't like it. A clean Jupyter interface does feel great though!

If you don't just want to read the notebook, but execute its cells yourself, all requirements in `requirements.txt` have to be installed. You don't have to do this externally! Just execute the installation cell at the top of notebook after starting it – you should execute the notebook from top to bottom anyway.

This assumes you have Python 3 and Jupyter already installed, of course.

Finally, open the notebook:

```
jupyter notebook Report.ipynb
```

## ⚠️ License
This repository has been published under the MIT license (see the file `LICENSE.txt`).

The [photo of the race car](https://unsplash.com/photos/XtVl8IL-8EI) has been shot by [Wes Tindel](https://unsplash.com/@lonestarexotic) and was published under the Unsplash license.
