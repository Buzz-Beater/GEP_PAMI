"""
Created on 4/20/19

@author: Baoxiong Jia

Description:

"""
from basemeta import Metadata
class BREAKFAST_METADATA(Metadata):
    def __init__(self):
        super(BREAKFAST_METADATA, self).__init__()

        self.activities = [
                                "salat", "tea", "coffee", "scrambledegg", "pancake",
                                "sandwich", "milk", "cereals", "friedegg", "juice"
                            ]

        self.subactivities = [
                                    "fry_egg", "add_saltnpepper", "cut_fruit", "pour_milk", "take_cup", "pour_water",
                                    "spoon_flour", "SIL", "stir_coffee", "pour_cereals", "butter_pan", "put_egg2plate",
                                    "take_glass", "pour_sugar", "stir_milk", "take_butter", "peel_fruit", "take_knife",
                                    "stirfry_egg", "pour_oil", "pour_flour", "spoon_powder", "put_pancake2plate",
                                    "stir_fruit", "squeeze_orange", "fry_pancake", "pour_dough2pan", "put_fruit2bowl",
                                    "stir_egg", "take_eggs", "put_bunTogether", "pour_coffee", "smear_butter",
                                    "cut_orange", "take_bowl", "cut_bun", "stir_tea", "take_squeezer", "pour_juice",
                                    "stir_cereals", "pour_egg2pan", "take_topping", "add_teabag", "crack_egg",
                                    "take_plate", "put_toppingOnTop", "stir_dough", "spoon_sugar"
                                ]

        self.actions = [
                            "fry_egg", "add_saltnpepper", "cut_fruit", "pour_milk", "take_cup", "pour_water",
                            "spoon_flour", "SIL", "stir_coffee", "pour_cereals", "butter_pan", "put_egg2plate",
                            "take_glass", "pour_sugar", "stir_milk", "take_butter", "peel_fruit", "take_knife",
                            "stirfry_egg", "pour_oil", "pour_flour", "spoon_powder", "put_pancake2plate",
                            "stir_fruit", "squeeze_orange", "fry_pancake", "pour_dough2pan", "put_fruit2bowl",
                            "stir_egg", "take_eggs", "put_bunTogether", "pour_coffee", "smear_butter",
                            "cut_orange", "take_bowl", "cut_bun", "stir_tea", "take_squeezer", "pour_juice",
                            "stir_cereals", "pour_egg2pan", "take_topping", "add_teabag", "crack_egg",
                            "take_plate", "put_toppingOnTop", "stir_dough", "spoon_sugar"
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