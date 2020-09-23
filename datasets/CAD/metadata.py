"""
Created on 11/27/18

@author: Baoxiong Jia

Description:

"""
from basemeta import Metadata
class CAD_METADATA(Metadata):
    def __init__(self):
        super(CAD_METADATA, self).__init__()

        self.activities = [
                            'arranging_objects', 'picking_objects', 'taking_medicine',
                            'making_cereal', 'cleaning_objects', 'stacking_objects', 'having_meal',
                            'microwaving_food', 'unstacking_objects', 'taking_food'
                           ]

        self.subactivities = [
                                'reaching', 'moving', 'pouring', 'eating', 'drinking',
                                'opening', 'placing', 'closing', 'null', 'cleaning', 'prior'
                            ]

        self.actions = [
                            'reaching', 'moving', 'pouring', 'eating', 'drinking',
                            'opening', 'placing', 'closing', 'null', 'cleaning'
                        ]

        self.objects = ['medcinebox', 'cup', 'bowl', 'box', 'milk', 'book', 'microwave', 'plate', 'remote', 'cloth']

        self.affordances = [
                        'movable', 'stationary', 'reachable', 'pourable', 'pourto', 'containable',
                        'drinkable', 'openable', 'placeable', 'closeable', 'cleanable', 'cleaner'
                        ]

        for a in self.activities:
            self.activity_index[a] = self.activities.index(a)

        for s in self.subactivities:
            self.subactivity_index[s] = self.subactivities.index(s)

        for a in self.actions:
            self.action_index[a] = self.actions.index(a)

        for o in self.objects:
            self.object_index[o] = self.objects.index(o)

        for u in self.affordances:
            self.affordance_index[u] = self.affordances.index(u)

        self.ACTIVITY_NUM = len(self.activities)
        self.SUBACTIVITY_NUM = len(self.subactivities)
        self.ACTION_NUM = len(self.actions)
        self.OBJECT_NUM = len(self.objects)
        self.AFFORDANCE_NUM = len(self.affordances)
