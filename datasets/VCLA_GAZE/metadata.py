"""
Created on 10/18/18

@author: Baoxiong Jia

Description:

"""
from basemeta import Metadata
class VCLA_METADATA(Metadata):
    def __init__(self):
        super(VCLA_METADATA, self).__init__()
        self.activities = [
                            'c01_sweep_floor', 'c02_mop_floor', 'c03_write_on_blackboard',
                            'c04_clean_blackboard', 'c05_use_elevator', 'c06_pour_liquid_from_jug',
                            'c07_make_coffee', 'c08_read_book', 'c09_throw_trash',
                            'c10_heat_food_with_microwave', 'c11_use_computer', 'c12_search_drawer',
                            'c13_move_bottle_to_dispenser', 'c14_open_door'
                        ]

        self.subactivities = [
                                'null',
                                'search', 'tear', 'read', 'throw', 'wring', 'open', 'use', 'walk', 'scrub', 'pour',
                                'write', 'sweep', 'grab', 'mop', 'close', 'push', 'stand', 'sit', 'grag', 'place', 'prior'
                            ]

        self.actions = [
                            'null',
                            'search', 'tear', 'read', 'throw', 'wring', 'open', 'use', 'walk', 'scrub', 'pour',
                            'write', 'sweep', 'grab', 'mop', 'close', 'push', 'stand', 'sit', 'grag', 'place'
                        ]

        self.objects = [
                            'null',
                            'blackboard', 'chair', 'dispenser', 'dustpan', 'eraser', 'cup', 'drawer',
                            'bucket', 'microwave', 'broom', 'button', 'handle', 'paper', 'door', 'mop',
                            'jug', 'bottle', 'monitor', 'book', 'food', 'can', 'chalk'
                        ]

        self.affordances = [
                                'null',
                                'usable', 'scrubber', 'searchable', 'wringable', 'scrubbable',
                                'throwable', 'sittable', 'sweepable', 'pourable', 'pourto', 'writer',
                                'writable', 'tearable', 'moppable', 'closeable', 'statuibar', 'placeable',
                                'stationary', 'readable', 'grabbable', 'openable', 'pushable'
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
        self.MAXIMUM_OBJ_VIDEO = 3

