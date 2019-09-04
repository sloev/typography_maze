# make typographic art from books

uses sentiment of words to make turns

## dependencies

```
python 3+

brew reinstall cairo pkg-config

pip install cairocffi afinn

npm install svgexport -g

```

## usage

```
# get a text file with lots of text inside and run

python main.py input.txt

# now you have a huge output.svg
# create a portable downscaled PNG with correct ratio and max 1000 in width

svgexport output.svg output2.png 1000:
```

## here is Kafkas *The Trial* as a typographic maze:

![kafkas The Trial](/the_trial.png?raw=true "Kafkas The Trial")

And here is an excerpt showing that the lines actually are text:

![kafkas The Trial](/the_trial_excerpt.svg?raw=true&sanitize=true "Kafkas The Trial excerpt")

