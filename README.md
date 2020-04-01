# Data Science Algorithm Gallery
Welcome to the Data Science Algorithm Gallery!

This repository demonstrates the inner workings of algorithms that are commonly used in Data Science.
It is comprised mostly of Jupyter notebooks that present the algorithms in both theoretical setting as well as some practical cases.
For the sake of the demonstration, each algorithm is implemented using Python librariessuch as `numpy`, `scipy` and `pandas`.
Then, the algorithms are compared to implementations using dedicated libraries (e.g. `scikit-learn`) whenever possible.

The main (and only) language used is Python 3.6.


## Content
Each algorithm has a dedicated directory that is complimented with a separate README page with a simple explanation and mathematical context.
Under each directory, some additional code may exists in order to support the notebooks.
The repository does not contain datasets, but the individual README files contain descriptions of steps to be taken to get the datasets.

## Installation
If you would like to start or verify the notebooks, use the virtual environment.
```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
cd ./<some-algorithm>
jupyter notebook
```

## Contribution
Please feel free to clone or fork this repository.
You are free to use is as you please, although I would feel appreciated if you mention or cite this work.

You are also welcome to contribute to this repository.
In this case, I am asking you to:
1. Create a separate directory for the algoritm.
2. Supplement it with a README file that introduces the algorithm at a mathematical level, and describes how to get the datasets if applicable.
3. Implement the algorithm and compare it against existing packages whenever applicable.


## Next steps
Creating of these notebooks is a lot of fun.
Although we did our best to present the content in a correct and structured manner, this is mainly a curousity-driven work and its main purpose is self-improvement and fun.
If you stumble across any inconsistency or would like suggest the direction of the future steps, please feel free to contact me on oleg@zerowithdot.com.

Hope you enjoy it as much as you find it useful.
Please, visit me at [zero-with-dot](https://zerowithdot.com/).

