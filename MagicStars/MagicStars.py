########################################################################
#                                                                      #
#   MagicStars.py                                                      #
#   09/07/2013                                                         #
#                                                                      #
#   I will scrape the Gatherer webpage to obtain the star ratings of   #
#   cards in my library, and probably add them in as a value in the    #
#   CSV file.  Maybe I can then port them back into MWS?               #
#                                                                      #
########################################################################

import sys
import os
import re
import copy
import itertools
from urllib import urlopen, quote_plus
from BeautifulSoup import BeautifulSoup



#----------------------------------------------------------------------#
#                                                                      #
#   Module Constants                                                   #
#                                                                      #
#----------------------------------------------------------------------#

MAGIC_BASE_URL = 'http://gatherer.wizards.com/Pages/Search/Default.aspx?name=+[]'
CSV_DATA_PATH = 'ZachLibrary_new.csv'
CSV_DATA_PATH_UPDATED = 'ZachLibrary_updated.csv'


#----------------------------------------------------------------------#
#                                                                      #
#   Module Routines                                                    #
#                                                                      #
#----------------------------------------------------------------------#

class MagicStars():

    
    def __init__( self ):
        self.loadCSVData()
    
    
    #
    #   Top-level functions
    #
    
    def loadCSVData( self ):
        """
        load the csv data we will be updating into an appropriate object
        saved as self.cardInventory
        """
        with open( CSV_DATA_PATH, 'r' ) as f:
            s = f.read()
        
        s = s.replace( '\r\n', '\n' )
        s = s.split( '\n' )
        
        self.cardInventory = {}
        
        for (i, card) in enumerate( s ):
            if i != 0:
                firstColon = card.find( ';' )
                name = card[ :firstColon ]
                rest = card[ firstColon: ]
                self.cardInventory[ name ] = rest


    def findAllStars( self ):
        """
        Cycle through the cards I own and find their star rating at
        Gatherer
        """
        for card in self.cardInventory:
            self.findCardStar( card )
        
        self.updatedCSVData()


    def findCardStar( self, card ):
        """
        Open the appropriate search URL, follow it to the correct card,
        parse it for the star rating.
        """
        try:
            page = self.properPage( card )
            rating = self.parseForRating( page )
            self.addRating( card, rating )
            print '{:<40}Rating {:<6}'.format( card, rating )
        except ValueError:
            self.cantFindCard( card )
    
    
    def addRating( self, card, rating ):
        """
        Add a rating to the self.cardInventory object
        """
        self.cardInventory[ card ] += '";"' + rating + '"'
    
    
    def cantFindCard( self, card ):
        """
        Indicate that we weren't able to find the proper URL, and might
        have to do it by hand
        """
        print "Can't find the info for card " + card
        self.addRating( card, '0.000' )
        

    def updatedCSVData( self ):
        """
        Save an updated CSV data file with the star ratings included
        """
        s = '"Name";"Edition";"Qty";"Rarity";"Color";"Cost";"Type";"P/T";"Text";"Comment";"Stars"'
        
        for ( name, rest ) in self.cardInventory.iteritems():
            s += '\r\n"' + name + rest
        
        with open( CSV_DATA_PATH_UPDATED, 'w' ) as f:
            f.write( s )
    
    
    #
    #   URL navigation and parsing functions
    #
    
    
    def properPage( self, card ):
        """
        Find the proper page url and return a BeautifulSoup object we
        can navigate
        """
        
        urlString = self.generateURL( card )
        searchPage = self.openSearchURL( urlString, card )
        return searchPage
    
    
    def generateURL( self, cardName ):
        """
        Create the text string of the URL we will open
        """
        
        newString = '[' + cardName + ']'
        
        #   fuse cards
        if newString.count( '/' ) != 0:
            newString = newString.replace( '/', ']+[//]+[' )
        
        urlString = copy.copy( MAGIC_BASE_URL )
        urlString = urlString.replace( '[]', newString )
        return urlString
    
    
    def openSearchURL( self, urlString, cardName ):
        """
        Open the URL and return a BeautifulSoup object
        """
        f = urlopen( urlString )
        
        massage = copy.copy( BeautifulSoup.MARKUP_MASSAGE )
        # Fix broken alt text (alt='Urza's Legacy')
        massage.extend( [ ( re.compile( r"'(Urza's.+?)'" ), lambda m: '"%s"' % m.group(1) ), ] )
        soup = BeautifulSoup( f, markupMassage = massage )
        
        if self.weFoundIt( soup ):
            return soup
        else:
            soup = self.keepLooking( soup, cardName )
        
        return soup
    
    
    def weFoundIt( self, searchPage ):
        """
        Check whether or not this is the correct (i.e. final) webpage
        """
        return searchPage.find( 'span', attrs = { 'class' : 'textRatingValue' } ) != None
    
    
    def keepLooking( self, searchPage, cardName ):
        """
        Find out which of the search objects is correct (if we can) and
        open / return the final page
        """
        
        #   Find the list of possible cards, check to see if any are
        #   exactly right
        tagList = searchPage.findAll( 'a', attrs = { 'onclick' : 'return CardLinkAction(event, this, \'SameWindow\');'} )
        
        for tag in tagList:
            s = str( tag.text )
            
            compString = str( cardName )
            
            #   fuse cards
            if compString.count( '/' ) != 0:
                compString = compString.replace( '/', ' // ' )
                compString += ' (' + compString[ :compString.find( ' // ' ) ] + ')'
            
            if s.lower() == compString.lower():
                urlString = str( tag.attrMap[u'href'] )
                urlString = urlString.replace( '..', 'http://gatherer.wizards.com/Pages' )
                return self.openSearchURL( urlString, cardName )
        
        with open( 'test.html', 'w' ) as f:
            f.write( searchPage.__repr__() )
        raise ValueError, "We got lost!"
    
    
    def parseForRating( self, page ):
        """
        Given a BeautifulSoup object generated by properPage, return the
        star rating assigned by the community at Gatherer (as a string)
        """
        x = str( page.find( 'span', attrs = { 'class' : 'textRatingValue' }) )
        rightCarat = x.find( '>' )
        leftCarat = x.find( '<', rightCarat )
        return x[ rightCarat + 1 : leftCarat ]
