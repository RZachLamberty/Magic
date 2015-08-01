#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module: MagicMana.py
Author: zlamberty
Created: 2013-09-07

Description:
    Monte Carlo simulation of the number of cards I can access as a function of
    turns

Usage:
    <usage>

"""

import copy
import cPickle as pickle
import itertools
import logging
import logging.config
import numpy as np
import os
import pandas as pd
import pylab
import scipy
import sys
import time
import yaml

from constants import _COLORS, _MANA_TYPES
from magichistory import History
from math import isnan


# ----------------------------- #
#   Module Constants            #
# ----------------------------- #

logger = logging.getLogger("MagicMana.py")
_HERE = os.path.dirname(os.path.realpath(__file__))
_LOGCONF = os.path.join(_HERE, 'logging.yaml')
with open(_LOGCONF, 'rb') as f:
    logging.config.dictConfig(yaml.load(f))


# ----------------------------- #
#   load decks                  #
# ----------------------------- #

def load_deck_from_text(fdeck):
    """ load the deck as it's laid out in the text with variable number of
        mana. Add in mana in accordance with the number in each color

    """
    return pd.read_csv(fdeck)


def add_mana(deck, white=0, blue=0, black=0, red=0, green=0, colorless=0):
    deck = pd.concat([
        deck,
        pd.DataFrame([{'name': 'white mana'} for i in range(white)]),
        pd.DataFrame([{'name': 'blue mana'} for i in range(blue)]),
        pd.DataFrame([{'name': 'black mana'} for i in range(black)]),
        pd.DataFrame([{'name': 'red mana'} for i in range(red)]),
        pd.DataFrame([{'name': 'green mana'} for i in range(green)]),
        pd.DataFrame([{'name': 'colorless mana'} for i in range(colorless)]),
    ])
    deck.index = range(deck.shape[0])
    return deck


# ----------------------------- #
#   configuring simulations     #
# ----------------------------- #

def load_configs(fyaml):
    """ yaml config file for all parameters """
    with open(fyaml, 'rb') as f:
        return yaml.load(f)


def hist_file_from_yaml(fyaml):
    yamlbase = os.path.splitext(os.path.basename(fyaml))[0]
    return os.path.join(_HERE, 'data', '{}.pkl'.format(yamlbase))


# ----------------------------- #
#   monte carlo hand sims       #
# ----------------------------- #

def run_simulation_from_yaml(fyaml):
    """ load the parameters of the simulation from the yaml and run """
    params = load_configs(fyaml)

    # all params values except seed can go straight into full_monte
    seed = params.pop('seed')
    np.random.seed(seed)
    history = full_monte(**params)
    history.save(hist_file_from_yaml(fyaml))


def full_monte(fdeck, manaRange, numGames, roundCutoff=None):
    """ Yurp """
    i = 1
    nomanadeck = load_deck_from_text(fdeck)
    history = History()
    mo = mana_options(manaRange)
    for (deckNum, (white, blue, black, red, green, colorless)) in mo:
        logger.info("Simulating game with the following mana pool:")
        logger.debug("white:     {}".format(white))
        logger.debug("blue:      {}".format(blue))
        logger.debug("black:     {}".format(black))
        logger.debug("red:       {}".format(red))
        logger.debug("green:     {}".format(green))
        logger.debug("colorless: {}".format(colorless))
        deck = add_mana(
            nomanadeck.copy(),
            white, blue, black, red, green, colorless
        )
        history.register_deck(deckNum, deck)
        simulate_games(numGames, deck, history, roundCutoff)

    #wrap_up_simulations()
    return history


def simulate_games(numGames, deck, history, roundCutoff=None):
    """ Given a deck, simulate NUMBER_OF_GAMES independent games. Start each
        by shuffelling, and play through the entire deck. Determine how many
        cards could be played given the mana drawn up to that point.

    """
    for gameIndex in range(numGames):
        progress_bar(gameIndex, numGames)
        deck = shuffle(deck)
        history.register_game(gameIndex)
        simulate_game(deck, history, roundCutoff)


def progress_bar(gameIndex, numGames):
    # 5% of games
    fivep = int(float(numGames) / 20)
    if fivep == 0 or gameIndex % fivep == 0:
        logger.info("{:>6.2f}% of games finished ({}/{})".format(
            100.0 * float(gameIndex) / numGames,
            gameIndex,
            numGames
        ))


def shuffle(deck):
    return deck.reindex(np.random.permutation(deck.index))


def simulate_game(deck, history, roundCutoff=None):
    """ Given the shuffled deck, simulate the draw. At each step, put in a mana
        if there is one and play any cards we can with all of the mana in our
        pool. Obviously this is imperfect, but it'll give a good idea of how
        many non-mana cards we can get out at a given round for an average
        draw.

    """
    currentHand = deck.head(7)
    manaPool = pd.DataFrame(
        data={'num': 0.0, 'untapped': 0.0},
        index=_MANA_TYPES
    )

    for round in range(7, roundCutoff or len(deck)):
        #logger.debug('round {}'.format(round - 7))
        currentHand = draw_card(currentHand, deck.iloc[round])
        currentHand, manaPool = play_mana(currentHand, manaPool)
        currentHand, manaPool, playedCards = play_from_hand(currentHand, manaPool)
        history.record_round(currentHand, manaPool, playedCards, round - 7)


def draw_card(currentHand, card):
    """ Draw a card and add it to your hand """
    return currentHand.append(card)


# ----------------------------- #
#   mana functions              #
# ----------------------------- #

def mana_options(manaRange={'red': (9, 13), 'blue': (9, 13), 'colorless': (0, 3)}):
    """ create an iterable which covers all options of the five colors given a
        dict with color: (low, high) bound key/vals

    """
    return enumerate(itertools.product(
        range(*manaRange.get('white', (0, 1))),
        range(*manaRange.get('blue', (0, 1))),
        range(*manaRange.get('black', (0, 1))),
        range(*manaRange.get('red', (0, 1))),
        range(*manaRange.get('green', (0, 1))),
        range(*manaRange.get('colorless', (0, 1))),
    ))


def play_mana(currentHand, manaPool):
    """ Add mana from your hand if you can (check to make sure you're playing
        the right mana color). return the updated hand (less mana card) and
        mana pool (plus mana color)

    """
    # Calculate the mana in the hand
    manaInHand = mana_in_hand(currentHand)

    # Pick one and play it
    manaColor = choose_mana(currentHand, manaPool, manaInHand)

    if manaColor is None:
        pass
    else:
        mcName = '{} mana'.format(manaColor)
        ci = currentHand[currentHand.name == mcName].index[0]
        currentHand = currentHand.drop(ci)

        manaPool.loc[manaColor, :] += 1.0

    return currentHand, manaPool


def mana_in_hand(currentHand):
    """ Calculate the mana in the current hand and return a df of it """
    mc = currentHand.name[currentHand.name.str.contains('mana')]
    mc = mc.str.replace(' mana', '')
    mc = mc.value_counts()
    return mc


def choose_mana(currentHand, manaPool, manaInHand):
    """ Calculate the net weight of the mana in my current hand based on the
        non-mana cards in that hand.

    """
    # Is there any mana at all, only one, or more than one?
    if len(manaInHand) == 0:
        manaColor = None
    # if we have only one color, return one of those
    elif len(manaInHand) == 1:
        manaColor = manaInHand.keys()[0]
    else:
        # we have multiple mana options. Which one gets us closest to playing a
        # card? Note: this will always play a colored mana if one exists; if
        # one does not then we will have been routed to the elif statement
        # above
        manaWeights = mana_color_weight(currentHand, manaPool)
        manaColor = max(
            [_ for _ in manaInHand.index if _ != 'colorless'],
            key=lambda c: manaWeights[c]
         )

    return manaColor


def mana_color_weight(currentHand, manaPool):
    """ We have some mana already in manaPool. Calculate the "net need" of each
        mana color, given the non-mana cards in our current hand. Give
        deference to those cards which we could cast immediately, even if the
        mana is a color that is not what the bulk of our future casts will
        require. Nothing special currently about what type of card we would
        cast.

    """
    nonmana = currentHand[currentHand.total.notnull()]

    cardCosts = {}
    maxTotal = 0

    # find the cost of each card less the current mana in manapool
    # (so manapool should probably be a pandas df too)
    cardcosts = nonmana.apply(func=smart_cost, axis=1, manaPool=manaPool)

    # weight each possible contribution by the ability of that single card to
    # cast the card (this will be done via an inverse exponential waiting of
    # the number of cards yet required to cast that particular spell)
    wt = 2 ** (cardcosts.total.max() - cardcosts.total)
    return cardcosts[_COLORS].mul(wt, axis=0).sum()


def smart_cost(card, manaPool):
    """ calculate the cost of the card assuming you use your mana pool """
    manacosts = card[_MANA_TYPES][card[_MANA_TYPES] > 0]
    tmpPool = use_mana_for_card(manacosts, manaPool.copy())

    # finally, add a 'total' row so that the returned series has the total
    # still needed as a value
    tmpPool.loc['total'] = tmpPool.sum()

    return tmpPool.needed


# ----------------------------- #
#   non-mana functions          #
# ----------------------------- #

def play_from_hand(currentHand, manaPool):
    """ Play any non-mana cards you can

        for each N and for each N-card combo, determine if it is playable, and
        if so what its total offense contribution would be. For each N, retain
        the combo with the highest offense

        if there is no N = 1 playable option, just return

        if we go through an entire N > 1 without a playable set, play the last
        highest combo and return

    """
    nonmana = currentHand[currentHand.total.notnull()]

    bestComboInd = []
    playedCards = None
    maxOff = -float('inf')

    for playableCombo in playable_combos(nonmana, manaPool):
        cards = nonmana.loc[playableCombo]
        offense = cards.off.sum()
        bestOffYet = (
            (offense > maxOff) or (maxOff == -float('inf') and isnan(offense))
        )
        if bestOffYet:
            maxOff = offense
            bestComboInd = playableCombo

    if bestComboInd:
        currentHand, manaPool, playedCards = play_combo(
            currentHand, bestComboInd, manaPool
        )

    return currentHand, manaPool, playedCards


def playable_combos(nonmana, manaPool):
    """ a shell-type iterator of cards in nonmana """
    # build / yield single card shell first
    shell = [
        [i] for i in nonmana.index
        if combo_is_playable(nonmana.loc[[i]], manaPool)
    ]
    for x in shell:
        yield x

    # now we iterate
    while shell:
        newShell = []
        for comboIndex in shell:
            for newIndex in nonmana.index:
                if newIndex not in comboIndex:
                    candidate = comboIndex + [newIndex]
                    cards = nonmana.loc[candidate]
                    if combo_is_playable(cards, manaPool):
                        newShell.append(candidate)
                        yield candidate
        shell = newShell


def card_combinations(nonmana, n):
    """ return an iterator of all non-repeating combo sets of n cards from
        withing nonmana

    """
    return itertools.combinations(nonmana.index, n)


def combo_is_playable(cards, manaPool):
    """ given a df of cards and a df of the manaPool, check to see if the cards
        can be played collectively

    """
    # convert to one composite card
    compCard = cards[_MANA_TYPES].sum()
    return is_playable(compCard, manaPool)


def is_playable(card, manaPool):
    # the card is playable iff the needed column sums to 0 after payment
    tmpPool = use_mana_for_card(card, manaPool.copy())
    return tmpPool.needed.sum() == 0


def play_combo(currentHand, indices, manaPool):
    ind = list(indices)
    cards = currentHand.loc[ind]

    # spend mana
    compCard = cards[_MANA_TYPES].sum()
    manaPool = use_mana_for_card(compCard, manaPool)[['num', 'untapped']]

    # put on table
    currentHand = currentHand.drop(ind)

    return currentHand, manaPool, cards


def use_mana_for_card(card, manaPool):
    manaPool['needed'] = card

    # tap mana color-to-color first (not just colored mana -- burn colorless
    # mana we have on needed colorless mana)
    totap = manaPool[['untapped', 'needed']].min(axis=1)
    manaPool.untapped -= totap
    manaPool.needed -= totap

    # remove as many colorless as are remaining; don't worry about tapping
    notTapped = manaPool.untapped.sum()
    totapColorless = min(notTapped, manaPool.needed.loc['colorless'])
    manaPool.needed.loc['colorless'] -= totapColorless
    yetToTapColorless = totapColorless
    while yetToTapColorless:
        firstInd = manaPool.untapped[manaPool.untapped > 0].index[0]
        manaPool.loc[firstInd, 'untapped'] -= 1.0
        yetToTapColorless -= 1
    return manaPool


#----------------------------------------------------------------------#
#                                                                      #
#   Monte Carlo hand simulations                                       #
#                                                                      #
#----------------------------------------------------------------------#

def comparisonDic(johnTrial = False):
    """
    For all of the files generated, create graphs comparing the number
    of cards and the size of the mana pool as a function of turn number
    """
    if johnTrial:
        simSetList = sorted([f.split('_')[2:] for f in os.listdir('./') if f.startswith('simulation_J')])
    else:
        simSetList = sorted([f.split('_')[1:] for f in os.listdir('./') if f.startswith('simulation_') and not f.startswith('simulation_J')])

    x = {}

    for simParams in simSetList:
        rS, bS = simParams
        r, b = int(rS), int(bS)

        if johnTrial:
            base = 'simulation_J_'
        else:
            base = 'simulation_'
        with open(base + rS + '_' + bS, 'rb') as f:
            simData = pickle.load(f)

        numGames, avMana, stdMana, avCards, stdCards = simData

        x[r, b] = [[ avMana,  stdMana], \
                      [avCards, stdCards]]

    return x


def comparisonPlot(x = None, anno = False, johnTrial = False):
    """
    Given a dictionary such as the one calculated above, plot all of
    the values.
    """
    if x == None:
        x = comparisonDic(johnTrial)

    #   Mana
    f1 = pylab.figure(1, figsize = (10,10))
    s1 = f1.add_subplot(111)
    #   Cards
    f2 = pylab.figure(2, figsize = (10,10))
    s2 = f2.add_subplot(111)

    for t in sorted(x.keys()):
        mSet, cSet = x[t]

        s1.errorbar(scipy.arange(len(mSet[0])), mSet[0], scipy.sqrt(mSet[1]), label = str(t))
        s2.errorbar(scipy.arange(len(cSet[0])), cSet[0], scipy.sqrt(cSet[1]), label = str(t))

    if anno:
        #   Make the annotation dictionary
        annoDic = {}
        xMax = 0
        for t in sorted(x.keys()):
            xMax = max(xMax, len(x[t][1][0]))
            yLast = round(x[t][1][0][-1], -1)
            if yLast not in annoDic:
                annoDic[yLast] = []
            annoDic[yLast].append(str(t))
        for yLast in annoDic:
            annoDic[yLast] = ','.join(annoDic[yLast])
            s2.annotate(annoDic[yLast], xy = (xMax, yLast), xytext = (0, yLast), arrowprops = dict(facecolor = 'black', shrink = 0.05, width = .1, headwidth = .4))

        #   Annotate

    f1.show()
    f2.show()


def bestHand(johnTrial = False):
    """
    For every number of steps, determine the sum difference through that
    round in number of cards or amount of mana available.
    """
    global HIGH_MANA_NUM, HIGH_MANA_NUM, COLORLESS_MANA, DECK_SIZE
    N = HIGH_MANA_NUM + HIGH_MANA_NUM + COLORLESS_MANA + DECK_SIZE
    x = comparisonDic(johnTrial)
    keyList = sorted(x.keys())
    L = len(keyList)
    cardAvArray = scipy.zeros(N)
    manaAvArray = scipy.zeros(N)
    numTrials = scipy.zeros(N)

    for (i, key) in enumerate(keyList):
        ll = len(x[key][0][0])
        manaAvArray[:ll] += x[key][0][0]
        cardAvArray[:ll] += x[key][1][0]
        numTrials[:ll] += 1

    manaAvArray /= numTrials
    cardAvArray /= numTrials

    delX = {}

    for key in keyList:
        delX[key] = copy.deepcopy(x[key])
        ll = len(delX[key][0][0])
        delX[key][0][0] -= manaAvArray[:ll]
        delX[key][1][0] -= cardAvArray[:ll]

    return delX


def cumulativeBestHand(johnTrial = False):
    """
    Basically the same as above, but instead of the difference from the
    mean, the integrated difference from the mean.
    """
    intDelX = {}
    delX = bestHand(johnTrial)

    intDelX = copy.deepcopy(delX)
    for key in sorted(delX.keys()):
        for i in range(len(delX[key][0][0])):
            for j in range(2):
                intDelX[key][j][0] = scipy.cumsum(delX[key][j][0])

    return intDelX


# ----------------------------- #
#   Main routine                #
# ----------------------------- #

def main():
    """ docstring """
    pass


# ----------------------------- #
#   Command line                #
# ----------------------------- #

def parse_args():
    """ Take a log file from the commmand line """
    parser = argparse.ArgumentParser()

    fyaml = "A yaml configuration file for a simulation to run"
    parser.add_argument("-f", "--fyaml", help=fyaml)

    args = parser.parse_args()

    logger.debug("arguments set to {}".format(vars(args)))

    return args


if __name__ == '__main__':

    args = parse_args()

    run_simulation_from_yaml(fyaml=args.fyaml)
