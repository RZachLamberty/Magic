# MTG Mana Analysis

I wrote this module in two phases: first, during a heavy mana playing phase toward the end of my graduate career (c. 2013), and second, in the summer of 2015. The first iteration was simple and sufficient, using a lot of dictionaries. The second version is a bit slicker, focusing instead on pandas dataframes. I had a fun time putting it together and using it, and I hope you will too.

There are two main things this package will do.

1. take a library of cards that you own (with a specific format, of course) and add the current "star ranking" for that card as compiled on the [gatherer website](http://gatherer.wizards.com/Pages/Card/Details.aspx?multiverseid=382866)
2. take a txt file specifying your current deck (sans mana) and some suggested options for which mana to put in with that deck, and output some metrics on the expected performance of those combinations.

The second is much more involved a process than the first, so I'll start with the easy one!

## Updating Star Ranking

I initially wrote this as part of a larger effort to document my mana collection. It is perhaps somewhat overkill, now that [this JSON dump](http://mtgjson.com/) of all magic cards exists, and is certainly easier to cross-reference against the gatherer site.

In any case, the process here is simple. Create a list of your cards as a csv, and just ensure that the csv itself has one column with heading "Name" and the name of the card for each row. The output file will be the same as the input file, but with a new column called "Stars."

### Usage
```bash
python magicstars.py -i data/ZachLibrary_new.csv -o data/ZachLibrary_updated.csv
```

## Mana Profiling

The general idea here is that you have a collection of cards (a deck) which you like to run together, and you want to know the appropriate amount of mana to use for that deck. The way I determine the proper mana configuration is by running several Monte Carlo simulations of possible deck shuffles / card draws, and calculating a few diagnostics.

The first step is to collect the cards. This is probably the most annoying step -- I hope to have something built at some point to do that in a more automated way, probably using that massive JSON dump I mentioned previously. In the meantime, however, I have taken two decks that my buddy John Mergo and I put together ages ago.

Once the deck is saved in a format that is similar to `data/deck.txt` (that is, with column names indicating the card's name, the breakdown of its mana cost (by color and total), and its offense / defense stats), you can either

1. run a simulation on that deck directly (using `full_monte(fdeck, manaRange, numGames, roundCutoff=None)`), or
2. create a `yaml` file specifying the configuration you expect to use for your simulation.

I highly recommend the latter!

### YAML config files

Making a `yaml` config file is easy! When making your own, you just need to have the following key/value pairs:

+ `fdeck`: the full path to the file name that describes your deck
+ `manaRange`: a dictionary of `[mana color]: [a, b]` pairs. The simulation will iterate over all possible combinations *including* `a` [mana color] cards up to (but *excluding*) `b` [mana color] cards. You can also add colorless mana with the key `colorless`
+ `numGames`: the number of independent games we should simulate for each deck + mana combination
+ `roundCutoff`: the point at which we should not draw any more cards / play any more mana, but just move on to the next game. Think of this as your "scoop" round -- if you haven't won by now, you ain't gonna
+ `seed`: a random number generator seed, to ensure reproducibility. 37 is very good for this. 42 is pretty decent as well.

One example file is `config/zachsmall.yaml`:
```yaml
fdeck: 'deck.txt'
manaRange:
  blue: [12, 14]
  red: [12, 14]
  colorless: [0, 3]
numGames: 30
roundCutoff: 20
seed: 42
```

With a config file such as the above fully specified, you can simply run

```bash
python magicmana.py --fyaml config/zachsmall.yaml
```

### Data analysis
The above is all well and good, but what are you going to do with it? Make plots, of course! From within python:

```python
import magichistory as H
import magicmana as M

fyaml = 'config/zachsmall.yaml'

history = H.History()
history.load(M.hist_file_from_yaml(fyaml))

history.mana_summary()
history.num_card_summary()
history.off_def_summary()
```

For a fuller description of the above, please see the [attached ipython notebook](https://github.com/RZachLamberty/Magic/blob/master/magic_mana_demo.ipynb).
