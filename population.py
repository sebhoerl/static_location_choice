import constant, utils
from tqdm import tqdm
import numpy as np
import xml.sax

class PopulationReader(xml.sax.ContentHandler):
    def __init__(self, config):
        self.progress = None
        self.config = config

        self.activities = []
        self.persons = []
        self.ids = []

        self.plan_selected = False
        self.leg_mode = None

    def startElement(self, name, attributes):
        self.progress.update()

        if name == "person":
            self.person_id = attributes['id']

        if name == "plan" and attributes["selected"] == "yes" and not ("freight_" in self.person_id or "cb_" in self.person_id):
            self.plan_selected = True
            self.persons.append(len(self.activities))
            self.ids.append(self.person_id)

        if name =="act" and self.plan_selected:
            self.activities.append((self.leg_mode, attributes['type'], attributes['facility']))

        if name == "leg" and self.plan_selected:
            self.leg_mode = attributes['mode']

    def endElement(self, name):
        if name == "plan": self.plan_selected = False

    def read(self, path, facility_id_to_index):
        cache = utils.load_cache("population", self.config)

        if cache is None:
            self.progress = tqdm(desc = "Loading Population ...")
            utils.make_xml_parser(self, utils.open_gzip(path))

            cache = self.process(facility_id_to_index)
            utils.save_cache("population", cache, self.config)
        else:
            print("Loaded population from cache.")

        return cache

    def process(self, facility_id_to_index):
        person_ids = self.ids

        person_indices = [(self.persons[i-1], self.persons[i]) for i in range(1, len(self.persons))]
        person_indices.append((self.persons[-1], len(self.persons)))

        activity_types = []
        activity_modes = []
        activity_facilities = []

        for activity in self.activities:
            activity_modes.append(constant.MODES_TO_INDEX[activity[0]] if activity[0] is not None else -1)
            activity_types.append(constant.ACTIVITY_TYPES_TO_INDEX[activity[1]])
            activity_facilities.append(facility_id_to_index[activity[2]])

        activity_types = np.array(activity_types, np.int)
        activity_modes = np.array(activity_modes, np.int)
        activity_facilities = np.array(activity_facilities, np.int)

        return person_ids, person_indices, activity_types, activity_modes, activity_facilities
