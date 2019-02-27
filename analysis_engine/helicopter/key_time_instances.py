import numpy as np
import six

from analysis_engine.node import (
    S, KeyTimeInstanceNode, helicopter_only
)


class EnterTransitionFlightToHover(KeyTimeInstanceNode):

    can_operate = helicopter_only

    def derive(self, holds=S('Transition Flight To Hover')):
        for hold in holds:
            self.create_kti(hold.slice.start)

class ExitTransitionFlightToHover(KeyTimeInstanceNode):

    can_operate = helicopter_only

    def derive(self, holds=S('Transition Flight To Hover')):
        for hold in holds:
            self.create_kti(hold.slice.stop)


class ExitTransitionHoverToFlight(KeyTimeInstanceNode):

    can_operate = helicopter_only

    def derive(self, holds=S('Transition Hover To Flight')):
        for hold in holds:
            self.create_kti(hold.slice.stop)
