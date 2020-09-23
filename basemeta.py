"""
Created on 10/18/18

@author: Baoxiong Jia

Description:

"""

class Metadata(object):
    def __init__(self):
        # list for constant strings
        self.activities = list()
        self.subactivities = list()
        self.actions = list()
        self.objects = list()
        self.affordances = list()

        # reverse index of strings
        self.activity_index = dict()
        self.subactivity_index = dict()
        self.action_index = dict()
        self.object_index = dict()
        self.affordance_index = dict()

        # Macro constant
        self.ACTIVITY_NUM = -1
        self.SUBACTIVITY_NUM = -1
        self.ACTION_NUM = -1
        self.OBJECT_NUM = -1
        self.AFFORDANCE_NUM = -1
        self.MAXIMUM_OBJ_VIDEO = -1