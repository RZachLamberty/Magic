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
    def __init__(self):
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
        manadf.loc[:, 'total'] = manadf[_MANA_TYPES].apply(np.sum, axis=1)
        grouped = manadf.groupby(['deck_num', 'round_num'])
        av, std = self.average_per_round(key=_MANA_TYPES + ['total'], df=manadf)

        color = _COLORS + ['gray', 'darkslategray', 'black']

        # multi-plot, faceted by deck nums
        deckNums = manadf.deck_num.unique()
        nrows, ncols = grid_layout_dimensions(len(deckNums))
        f, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=[16, 12])

        x = self.gameState.round_num.unique()
        ymax = av.total.max()

        for (i, j) in itertools.product(range(nrows), range(ncols)):
            dn = deckNums[i + j * nrows]

            try:
                ax = axes[i, j] if len(axes.shape) > 1 else axes[j]
            except AttributeError:
                ax = axes

            # add "ideal" mana curve as dashed line
            plt.sca(ax)
            plt.plot(x, x + 1, color='k', linestyle=":")

            # plot colors only first
            av.loc[dn].plot(yerr=std.loc[dn], color=color, ax=ax)

            # set constant range for all comparison graphs
            ax.set_ylim(0, ymax)

            # set the title
            t = self.get_deck_title(dn)
            ax.set_title(t)

        f.tight_layout()
        plt.show()
        plt.close(f)

    def num_card_summary(self):
        """ similar to the mana summary, but plotting the average number of
            cards played / cards in hand per round

        """
        avcardsplayed, acperr = self.average_per_round(key='cards_played')
        self.gameState.loc[:, 'cards_played_cumulative'] = self.cumsum_per_game(key='cards_played')
        cardsplayed, cperr = self.average_per_round(key='cards_played_cumulative')
        inhand, iherr = self.average_per_round(key='cards_in_hand')

        # multi-plot, faceted by deck nums
        deckNums = self.gameState.deck_num.unique()
        nrows, ncols = grid_layout_dimensions(len(deckNums))
        f, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=[16, 12])

        ymax = max(
            inhand.cards_in_hand.max(),
            cardsplayed.cards_played_cumulative.max()
        )

        for (i, j) in itertools.product(range(nrows), range(ncols)):
            dn = deckNums[i + j * nrows]

            try:
                ax = axes[i, j] if len(axes.shape) > 1 else axes[j]
            except AttributeError:
                ax = axes

            # bar chart of cards played per round
            avcardsplayed.loc[dn].plot(yerr=acperr.loc[dn], ax=ax, kind='bar')

            # line plot of cumulative cards played
            cardsplayed.loc[dn].plot(yerr=cperr.loc[dn], ax=ax)

            # line plot of cards in hand
            inhand.loc[dn].plot(yerr=iherr.loc[dn], ax=ax)

            # set constant range for all comparison graphs
            ax.set_ylim(0, ymax)

            # set the title
            t = self.get_deck_title(dn)
            ax.set_title(t)

        f.tight_layout()
        plt.show()
        plt.close(f)

    def off_def_summary(self):
        """ similar to the num card summary, but plotting the evolution of
            offense and defnese for played cards

        """
        self.gameState.loc[:, 'added_off_cumulative'] = self.cumsum_per_game(key='added_off')
        self.gameState.loc[:, 'added_def_cumulative'] = self.cumsum_per_game(key='added_def')
        offav, offerr = self.average_per_round(key='added_off_cumulative')
        defav, deferr = self.average_per_round(key='added_def_cumulative')

        # multi-plot, faceted by deck nums
        deckNums = self.gameState.deck_num.unique()
        nrows, ncols = grid_layout_dimensions(len(deckNums))
        f, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=[16, 12])

        ymax = max(
            offav.added_off_cumulative.max(),
            defav.added_def_cumulative.max()
        )

        for (i, j) in itertools.product(range(nrows), range(ncols)):
            dn = deckNums[i + j * nrows]

            try:
                ax = axes[i, j] if len(axes.shape) > 1 else axes[j]
            except AttributeError:
                ax = axes

            # line plot of cumulative off
            offav.loc[dn].plot(yerr=offerr.loc[dn], ax=ax)

            # line plot of cumulative def
            defav.loc[dn].plot(yerr=deferr.loc[dn], ax=ax)

            # set constant range for all comparison graphs
            ax.set_ylim(0, ymax)

            # set the title
            t = self.get_deck_title(dn)
            ax.set_title(t)

        f.tight_layout()
        plt.show()
        plt.close(f)

    def average_per_round(self, key, df=None):
        if not isinstance(key, list):
            keylist = [key]
        else:
            keylist = key
        if df is None:
            df = self.gameState
        grouped = df.loc[:, ['deck_num', 'round_num'] + keylist]
        grouped = grouped.groupby(['deck_num', 'round_num'])
        x = np.sqrt(len(self.gameState.game_num.unique()))
        return grouped.aggregate(np.average), grouped.aggregate(np.std) / x

    def cumsum_per_game(self, key):
        grouped = self.gameState.loc[:, ['deck_num', 'game_num', key]].fillna(0)
        grouped = grouped.groupby(['deck_num', 'game_num'])
        return grouped.cumsum()

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
            for k in ['deck_profile', 'game_state']
        }

        logger.debug('Saving to files: {}'.format(', '.join(fnames.values())))

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
            for k in ['deck_profile', 'game_state']
        }

        logger.debug(
            'Loading from files: {}'.format(', '.join(fnames.values()))
        )
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
