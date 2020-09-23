"""
Created on 11/27/18

@author: Baoxiong Jia

Description:

"""
from basemeta import Metadata

class WNP_METADATA(Metadata):
    def __init__(self):
        super(WNP_METADATA, self).__init__()

        self.activities = ['office', 'kitchen']
        self.subactivities = [
                                'null',
                                'fetch_from_fridge', 'put_back_to_fridge', 'prepare_food', 'microwaving', 'fetch_from_oven',
                                'pouring', 'drinking', 'leave_kitchen', 'fill_kettle', 'plug_in_kettle', 'move_kettle',
                                'reading', 'walking', 'leave_office', 'fetch_book', 'put_back_book', 'put_down_item',
                                'take_item', 'play_computer', 'turn_on_monitor', 'turn_off_monitor'
                            ]
        self.actions = [
                            'null',
                            'fetch_from_fridge', 'put_back_to_fridge', 'prepare_food', 'microwaving', 'fetch_from_oven',
                            'pouring', 'drinking', 'leave_kitchen', 'fill_kettle', 'plug_in_kettle', 'move_kettle',
                            'reading', 'walking', 'leave_office', 'fetch_book', 'put_back_book', 'put_down_item',
                            'take_item', 'play_computer', 'turn_on_monitor', 'turn_off_monitor'
                        ]

        for a in self.activities:
            self.activity_index[a] = self.activities.index(a)

        for s in self.subactivities:
            self.subactivity_index[s] = self.subactivities.index(s)

        for a in self.actions:
            self.action_index[a] = self.actions.index(a)

        self.ACTIVITY_NUM = len(self.activities)
        self.SUBACTIVITY_NUM = len(self.subactivities)
        self.ACTION_NUM = len(self.actions)


if __name__ == '__main__':
    metadata = WNP_METADATA()