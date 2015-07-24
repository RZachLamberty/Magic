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


# ----------------------------- #
#   Module Constants            #
# ----------------------------- #

_COLORS = ['white', 'blue', 'black', 'red', 'green', 'grey']
DECK_SIZE = 71 #45
RED_MANA = 13
BLUE_MANA = 13
GREY_MANA = 2
DTYPE = [('name', 'S10'), ('red', 'int32'), ('blue', 'int32'), ('grey', 'int32'), ('total', 'int32'), ('off', 'int32'), ('def', 'int32')]

DECK_FILE = 'johnDeck.txt' #deck.txt
SAVE_FILE_BASE = 'simulation_J_' #simulation_
LOW_MANA_NUM = 9 #11
HIGH_MANA_NUM = 12 #15

NUMBER_OF_GAMES = 10
CARDS_PLAYED_HISTORY = None
MANA_POOL_HISTORY = None
CARDS_PLAYED_2_HISTORY = None
MANA_POOL_2_HISTORY = None
TEMP_CARDS = None
TEMP_MANA = None

logger = logging.getLogger("MagicMana.py")
_LOGCONF = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), 'logging.yaml'
)
with open(_LOGCONF, 'rb') as f:
    logging.config.dictConfig(yaml.load(f))


# ----------------------------- #
#   load decks                  #
# ----------------------------- #

def load_deck_from_text(fdeck, white=0, blue=0, black=0, red=0, green=0, grey=0):
    """ load the deck as it's laid out in the text with variable number of
        mana. Add in mana in accordance with the number in each color

    """
    deck = pd.read_csv(fdeck)
    deck = pd.concat([
        deck,
        pd.DataFrame([{'name': 'white mana'} for i in range(white)]),
        pd.DataFrame([{'name': 'blue mana'} for i in range(blue)]),
        pd.DataFrame([{'name': 'black mana'} for i in range(black)]),
        pd.DataFrame([{'name': 'red mana'} for i in range(red)]),
        pd.DataFrame([{'name': 'green mana'} for i in range(green)])
        pd.DataFrame([{'name': 'grey mana'} for i in range(grey)]),
    ])
    deck.index = range(deck.shape[0])
    return deck


# ----------------------------- #
#   monte carlo hand sims       #
# ----------------------------- #

def full_monte(fdeck, manaRange, numGames):
    """ Yurp """
    i = 1
    for (white, blue, black, red, green, grey) in mana_options(manaRange):
        logger.info("Simulating game with the following mana pool:")
        logger.debug("white: {}".format(white))
        logger.debug("blue:  {}".format(blue))
        logger.debug("black: {}".format(black))
        logger.debug("red:   {}".format(red))
        logger.debug("green: {}".format(green))
        logger.debug("grey:  {}".format(green))
        deck = load_deck_from_text(fdeck, white, blue, black, red, green, grey)
        hand_simulations(numGames, deck)
        wrap_up_simulations()


def mana_options(manaRange={'red': (9, 13), 'blue': (9, 13), 'grey': (0, 3)}):
    """ create an iterable which covers all options of the five colors given a
        dict with color: (low, high) bound key/vals

    """
    return itertools.product(
        range(*manaRange.get('white', (0, 1))),
        range(*manaRange.get('blue', (0, 1))),
        range(*manaRange.get('black', (0, 1))),
        range(*manaRange.get('red', (0, 1))),
        range(*manaRange.get('green', (0, 1))),
        range(*manaRange.get('grey', (0, 1))),
    )


def hand_simulations(numGames, deck):
    """ Given the hand in deck, simulate NUMBER_OF_GAMES independent
        shuffled hands. Use each to determine how many cards could be
        played given the mana drawn up to that point.

    """
    for gameIndex in range(numGames):
        if gameIndex % 100 == 0:
            logger.info("{:>6.2f}% of games finished".format(
                100.0 * float(gameIndex) / numGames
            ))
        deck = shuffle(deck)
        simulate_game(deck)


def shuffle(deck):
    return deck.reindex(np.random.permutation(deck.index))


def simulate_game(deck):
    """ Given the shuffled deck, simulate the draw. At each step, put in a mana
        if there is one and play any cards we can with all of the mana in our
        pool. Obviously this is imperfect, but it'll give a good idea of how
        many non-mana cards we can get out at a given round for an average
        draw.

    """
    currentHand = deck.head(7)
    manaPool = {c: 0 for c in _COLORS}

    wipe_temp_deck()

    for i in range(7, len(deck)):
        currentHand = draw_card(currentHand, deck.iloc[i])
        currentHand = play_mana(currentHand, manaPool, i)
        playCards(currentHand, manaPool, i)

    updateFromTempDeck()


def draw_card(currentHand, card):
    """ Draw a card and add it to your hand """
    return currentHand.append(card)


def wrapUpSimulation():
    """
    do our calculations and dump the results to file
    """
    with open(SAVE_FILE_BASE + str(RED_MANA) + '_' + str(BLUE_MANA), 'wb') as f:
        avMana = MANA_POOL_HISTORY / NUMBER_OF_GAMES
        avCards = CARDS_PLAYED_HISTORY / NUMBER_OF_GAMES
        stdMana = MANA_POOL_2_HISTORY / (NUMBER_OF_GAMES - 1) - (NUMBER_OF_GAMES / (NUMBER_OF_GAMES - 1)) * avMana**2
        stdCards = CARDS_PLAYED_2_HISTORY / (NUMBER_OF_GAMES - 1) - (NUMBER_OF_GAMES / (NUMBER_OF_GAMES - 1)) * avCards**2
        dumpTup = (NUMBER_OF_GAMES, avMana, stdMana, avCards, stdCards)
        pickle.dump(dumpTup, f)


#   Utilities   -------------------------------------------------------#
def wipe_temp_deck():
    """ Clear out the global variables TEMP_* """
    global TEMP_MANA, TEMP_CARDS
    TEMP_MANA = scipy.zeros(DECK_SIZE + RED_MANA + BLUE_MANA + GREY_MANA)
    TEMP_CARDS = scipy.zeros(DECK_SIZE + RED_MANA + BLUE_MANA + GREY_MANA)


def updateFromTempDeck():
    """
    Use the information recorded in our temporary deck holders TEMP_*
    to update the running average and standard deviation dictionaries
    """
    global MANA_POOL_HISTORY, CARDS_PLAYED_HISTORY, MANA_POOL_2_HISTORY, CARDS_PLAYED_2_HISTORY, TEMP_CARDS, TEMP_MANA

    #   Make cumulative sums
    TEMP_MANA = scipy.cumsum(TEMP_MANA)
    TEMP_CARDS = scipy.cumsum(TEMP_CARDS)

    MANA_POOL_HISTORY += TEMP_MANA
    MANA_POOL_2_HISTORY += TEMP_MANA**2
    CARDS_PLAYED_HISTORY += TEMP_CARDS
    CARDS_PLAYED_2_HISTORY += TEMP_CARDS**2


# ----------------------------- #
#   mana functions              #
# ----------------------------- #

def play_mana(currentHand, manaPool, i):
    """ Add mana from your hand if you can (check to make sure you're playing
        the right mana color)

    """
    # Calculate the mana in the hand
    manaInHand = mana_from_hand(currentHand)

    # Pick one and play it
    manaColor = choose_mana(currentHand, manaPool, manaInHand)
    castMana(currentHand, manaPool, manaColor, i)


def mana_from_hand(currentHand):
    """ Calculate the mana in the current hand and return a dic of it """
    manaCount = currentHand.loc[currentHand.total.isnull(), 'name']
    manaCount = manaCount.value_counts()
    return {
        manaKey.replace(' mana', ''): ct
        for (manaKey, ct) in manaCount.iteritems()
    }


def choose_mana(currentHand, manaPool, manaInHand):
    """ Calculate the net weight of the mana in my current hand based on the
        non-mana cards in that hand

    """
    # Is there any mana at all, only one, or more than one?
    if len(manaInHand) == 0:
        manaColor = None
    elif len(manaInHand) == 1:
        manaColor = manaInHand.keys()[0]
    else:
        if 'grey' in manaInHand:
            manaColor = 'grey'
        else:
            manaWeights = mana_color_weight(currentHand, manaPool)

            if manaWeights['red'] >= manaWeights['blue']:
                manaColor = 'red'
            else:
                manaColor = 'blue'

    return manaColor


def mana_color_weight(currentHand, manaPool):
    """ Return a dictionary of the weights associated with colors in the mana
        pool

    """
    RED, BLUE, GREY = 1, 2, 3

    cardCosts = {}
    maxTotal = 0

    for card in currentHand:
        cardName = card[0]
        if cardName.find('mana') == -1:
            cardCosts[cardName] = [max(0, card[RED ] - manaPool['red' ]), \
                                      max(0, card[BLUE] - manaPool['blue']), \
                                      max(0, card[GREY ] - manaPool['grey' ]) ]
            tot = scipy.sum(cardCosts[cardName])
            cardCosts[cardName].append(tot)
            maxTotal = max(tot, maxTotal)

    redWeight, blueWeight = 0, 0
    for (card, weights) in cardCosts.iteritems():
        r, b, a, t = weights
        tenP = 10**(maxTotal - t)
        if a != 0:
            redWeight += tenP
            blueWeight += tenP
        else:
            if r != 0:
                redWeight += tenP
            if b != 0:
                blueWeight += tenP

    return {'red' : redWeight, 'blue' : blueWeight}


def castMana(currentHand, manaPool, manaColor, i):
    """
    Drop the mana of color manaColor from the currentHand, and add it to
    the manaPool
    """
    currentHandCopy = list(currentHand)
    if manaColor == None:
        pass
    else:
        notPlayed = True
        j = 0
        while j < len(currentHand) and notPlayed:
            card = currentHand[j]
            if card[0] == manaColor + ' mana':
                currentHand.pop(j)
                manaPool[manaColor] += 1
                notPlayed = False
            j += 1

        if notPlayed and j == len(currentHand) + 1:
            raise ValueError, "Mana of color " + manaColor + " didn't get played for some reason"

        TEMP_MANA[i - 7] += 1
        #raw_input('mana\n\t' + str(MANA_POOL_HISTORY[0,:]))



#   Play Cards  -------------------------------------------------------#

def playCards(currentHand, manaPool, i):
    """
    Play any cards you can
    """
    unspentMana = scipy.sum(manaPool.values())
    playableCards = singlePlayableCards(currentHand, manaPool, unspentMana)

    #   Walk through playable cards, play them in order
    canPlaySet = possiblePlays(playableCards, manaPool)

    if canPlaySet != [[]]:

        cardsToPlay = max(canPlaySet, key = lambda x : len(x))

        for card in cardsToPlay:
            playCard(card, currentHand, i)


def singlePlayableCards(currentHand, manaPool, unspentMana):
    return [card for card in currentHand if card[0].find('mana') == -1 and card[1] <= manaPool['red'] and card[2] <= manaPool['blue'] and card[4] <= unspentMana]


def possiblePlays(playableCards, manaPool):
    """
    return a list of all the collections of cards we can play given the
    mana we have
    """
    canPlaySet = [[]]
    for j in range(len(playableCards)):
        newCanPlaySet = []
        addedOption = False
        for previousCombo in canPlaySet:
            tempPlayable = [card for card in playableCards if card not in previousCombo]

            for card in tempPlayable:
                combo = previousCombo + [card]
                if canPlay(combo, manaPool):
                    newCanPlaySet.append(combo)
                    addedOption = True
        if not addedOption:
            break
        else:
            canPlaySet = newCanPlaySet

    return canPlaySet


def playCard(card, currentHand, i):
    """
    Play the card, indicate that we were able to play it in the global
    variable
    """
    currentHand.remove(card)
    TEMP_CARDS[i - 7] += 1
    #raw_input('played ' + card[0] + '\n\t' + str(CARDS_PLAYED_HISTORY[0,:]))


def canPlay(combo, manaPool):
    """
    See if we can play a combination of cards given the manapool we
    currently have
    """
    TOTAL_RED = manaPool['red']
    TOTAL_BLUE = manaPool['blue']
    EXTRA = manaPool['grey']
    TOTAL = TOTAL_RED + TOTAL_BLUE + EXTRA

    comboCost = scipy.sum([list(card)[1 : 4] for card in combo], axis = 0)

    comboSum = scipy.sum(comboCost)

    for card in combo:
        if (TOTAL >= comboSum):
            if (TOTAL_RED >= comboCost[0]):
                return (TOTAL_BLUE >= comboCost[1])
            else:
                return False
        else:
            return False


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
    global HIGH_MANA_NUM, HIGH_MANA_NUM, GREY_MANA, DECK_SIZE
    N = HIGH_MANA_NUM + HIGH_MANA_NUM + GREY_MANA + DECK_SIZE
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
    parser.add_argument("-x", "--xample", help="An Example", action='store_true')

    args = parser.parse_args()

    logger.debug("arguments set to {}".format(vars(args)))

    return args


if __name__ == '__main__':

    args = parse_args()

    main()
