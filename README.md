# Rant or Rave: Variation over Time in the Language of Online Reviews
Authors: Yftah Ziser, Bonnie Webber, and Shay Cohen (The University of Edinburgh).

This code repository generates results appearing in the paper ["Rant or Rave: Variation over Time in the Language of Online Reviews"](https://link.springer.com/article/10.1007/s10579-023-09652-5)

## Setting up the repository and the data
```bash
git clone https://github.com/yftah89/ReviewsOverTime
cd ReviewsOverTime
wget "https://bollin.inf.ed.ac.uk/public/direct/lre-data.zip"
unzip lre-data.zip -d data
rm lre-data.zip
```
For more information about the data structure, read data/README.md

## Creating a virtual environment 
```bash
python3 -m venv sot
source sot/bin/activate
pip3 install -r requirements.txt
```

## How to Cite Our Work
```
@article{ziser2023rant,
  title={Rant or rave: variation over time in the language of online reviews},
  author={Ziser, Yftah and Webber, Bonnie and Cohen, Shay B},
  journal={Language Resources and Evaluation},
  pages={1--31},
  year={2023},
  publisher={Springer}
}


If you use this implementation in your article, please cite: (add citation information).



