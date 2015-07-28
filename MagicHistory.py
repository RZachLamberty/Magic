#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module: MagicHistory.py
Author: zlamberty
Created: 2015-07-26

Description:
    Class for recording the history (cards played, manapool) of a magic game

Usage:
    <usage>

"""

import itertools
import logging
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns

from constants import _COLORS, _MANA_TYPES
from math import floor, isnan


# ----------------------------- #
#   Module Constants            #
# ----------------------------- #

logger = logging.getLogger(__name__)


# ----------------------------- #
#   Main class def              #
# ----------------------------- #

class History(object):
    """ this is a simple pandas df object with some utility functions """
    def __init__(self, numGames):
        self.numGames = numGames
        self.game = 0
        self.round = 0
        self.deckProfile = pd.DataFrame(
            columns=['deck_size'] + _MANA_TYPES
        )
        columns = ['deck_num', 'game_num', 'round_num']
        columns += _MANA_TYPES
        columns += ['cards_in_hand', 'cards_played', 'added_off', 'added_def']
        self.gameState = pd.DataFrame(columns=columns)

    def register_deck(self, deckNum, deck):
        """ one df should record the size of the deck and its mana contents """
        self.deckNum = deckNum
        ds = deck.shape[0]
        mc = deck.name[deck.name.str.endswith(' mana') & deck.total.isnull()]
        mc = mc.str.replace(' mana', '').value_counts()
        self.deckProfile.loc[deckNum] = pd.Series({'deck_size': ds}).append(mc)

    def register_game(self, gameNum):
        """ rather than carry the particular game index around, just add it as
            a member variable. This will be used as the index in the overall
            statistics df

        """
        self.gameNum = gameNum

    def record_round(self, currentHand, manaPool, playedCards, roundNum):
        state = pd.Series({
            'deck_num': self.deckNum,
            'game_num': self.gameNum,
            'round_num': roundNum,
        })
        state = state.append(manaPool.num)
        if playedCards is None:
            cp = 0
            ao = ad = float('nan')
        else:
            cp = playedCards.shape[0]
            ao = playedCards.off.sum()
            ad = playedCards['def'].sum()
        state = state.append(pd.Series({
            'cards_in_hand': currentHand.shape[0],
            'cards_played': cp,
            'added_off': ao,
            'added_def': ad,
        }))
        self.gameState = self.gameState.append(state, ignore_index=True)

    # plotting shiz ------------------------------------------------------------
    def mana_summary(self):
        """ for each mana type, and for total, produce a plot of manapool
            availability of colors as a function of time with std error bars

        """
        manadf = self.gameState[['deck_num', 'round_num'] + _MANA_TYPES]
        manadf.loc[:, 'total'] = manadf[_MANA_TYPES].apply(sum, axis=1)
        grouped = manadf.groupby(['deck_num', 'round_num'])
        av = grouped.aggregate(np.average)
        x = np.sqrt(len(self.gameState.game_num.unique()))
        std = grouped.aggregate(np.std) / x

        color = _COLORS + ['lightgray', 'gray']

        # multi-plot, faceted by deck nums
        deckNums = manadf.deck_num.unique()
        nrows, ncols = grid_layout_dimensions(len(deckNums))
        f, axes = plt.subplots(nrows=nrows, ncols=ncols)

        for (i, j) in itertools.product(range(nrows), range(ncols)):
            dn = deckNums[i + j * nrows]

            try:
                ax = axes[i, j] if len(axes.shape) > 1 else axes[j]
            except AttributeError:
                ax = axes

            av.loc[dn].plot(yerr=std.loc[dn], color=color, ax=ax)

            t = self.get_deck_title(dn)
            ax.set_title(t)

        plt.show()
        plt.close(f)

    def get_deck_title(self, dn):
        dn = int(dn)
        t = "Deck {}\n({})".format(
            int(dn),
            ', '.join(
                '{}: {:.0f}'.format(k, v)
                for (k, v) in self.deckProfile.loc[int(dn)].iteritems()
                if not isnan(v)
            )
        )
        return t

    # file io ------------------------------------------------------------------
    def save(self, fname):
        """ fname should correspond to a yaml file via
            config/name.yaml <--> data/name.pkl

            without pushing this into postgres, this is the best quick option for
            simultaneously keeping parameterization data around

        """
        fdir, fbase = os.path.split(fname)
        b, ext = os.path.splitext(fbase)

        fnames = {
            k: os.path.join(fdir, '{}.{}{}'.format(b, k, ext))
            for k in ['num_games', 'deck_profile', 'game_state']
        }

        logger.debug('Saving to files: {}'.format(', '.join(fnames.values())))
        with open(fnames['num_games'], 'wb') as f:
            f.write('{:.0f}'.format(self.numGames))

        self.deckProfile.to_pickle(fnames['deck_profile'])
        self.gameState.to_pickle(fnames['game_state'])

    def load(self, fname):
        """ fname should correspond to a yaml file via
            config/name.yaml <--> data/name.pkl

        """
        fdir, fbase = os.path.split(fname)
        b, ext = os.path.splitext(fbase)

        fnames = {
            k: os.path.join(fdir, '{}.{}{}'.format(b, k, ext))
            for k in ['num_games', 'deck_profile', 'game_state']
        }

        logger.debug(
            'Loading from files: {}'.format(', '.join(fnames.values()))
        )
        with open(fnames['num_games'], 'rb') as f:
            self.numGames = int(f.read().strip())

        self.deckProfile = pd.read_pickle(fnames['deck_profile'])
        self.gameState = pd.read_pickle(fnames['game_state'])

def grid_layout_dimensions(n):
    """ assuming we want to plot n items, what is the best way to lay them out
        in a i x j grid?

    """
    # first, find the smallest int larger than sqrt(n) by which n is evenly divisible
    for nrows in range(int(floor(np.sqrt(n))), 0, -1):
        if n % nrows == 0:
            ncols = n / nrows
            return nrows, ncols
    raise ValueError("Report this number to the nearest math authorities")
