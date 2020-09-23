"""
Created on 11/27/18

@author: Baoxiong Jia

Description:

"""

class SParseGraph(object):
    def __init__(self, start_frame, end_frame, subactivity=None, action=None, objects=list(), affordance_labels=list()):
        self._start_frame = start_frame
        self._end_frame = end_frame
        self._action = action
        self._subactivity = subactivity
        self._objects = objects
        self._affordance_labels = affordance_labels

        self._skeletons = None
        self._obj_positions = list()

    def __str__(self):
        return '{}-{} {} {} {}'.format(self._start_frame, self._end_frame, self._subactivity,
                                       self._objects, self._affordance_labels)

    def __repr__(self):
        return self.__str__()

    @property
    def id(self):
        return self._subactivity

    @property
    def subactivity(self):
        return self._subactivity

    @property
    def action(self):
        return self._action

    @property
    def start_frame(self):
        return self._start_frame

    @property
    def end_frame(self):
        return self._end_frame

    @property
    def objects(self):
        return self._objects

    @property
    def affordance(self):
        return self._affordance_labels

    @property
    def skeletons(self):
        return self._skeletons

    @property
    def obj_positions(self):
        return self._obj_positions

    def set_skeletons(self, skeletons):
        assert self._end_frame - self._start_frame + 1 == skeletons.shape[0]
        self._skeletons = skeletons

    def set_obj_positions(self, obj_positions):
        for obj in obj_positions:
            self._obj_positions.append(obj[self._start_frame : self._end_frame + 1])

    @subactivity.setter
    def subactivity(self, value):
        self._subactivity = value

    @end_frame.setter
    def end_frame(self, value):
        self._end_frame = value


class TParseGraph(object):
    def __init__(self, activity=None, sequence_id=None, subject=None):
        self._activity = activity
        self._sequence_id = sequence_id
        self._subject = subject
        self._terminals = list()

    def __str__(self):
        sequence = '{} * '.format(self._sequence_id)
        for t in self._terminals:
            sequence += t.id + ' '
        sequence += '#'
        return sequence

    def __repr__(self):
        return self.__str__()

    @property
    def activity(self):
        return self._activity

    @property
    def id(self):
        return self._sequence_id

    @property
    def subject(self):
        return self._subject

    @property
    def terminals(self):
        return self._terminals

    @property
    def length(self):
        if len(self._terminals) == 0:
            return 0
        else:
            return  self._terminals[-1].end_frame - self._terminals[0].start_frame + 1

    @property
    def start_frame(self):
        assert(len(self._terminals) > 0, 'No spg added')
        return self._terminals[0].start_frame

    @activity.setter
    def activity(self, value):
        self._activity = value

    def append_terminal(self, spg):
        self._terminals.append(spg)
