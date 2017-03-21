import constant, utils
from tqdm import tqdm
import numpy as np
import xml.sax
import re

TC = np.array((3600.0, 60.0, 1.0), dtype = np.float)
convert_time = lambda s: np.dot(TC, np.array(s.split(':'), dtype=np.float))

class PopulationReader(xml.sax.ContentHandler):
    def __init__(self, config):
        self.progress = None
        self.config = config

        self.activities = []
        self.persons = []
        self.ids = []

        self.plan_selected = False
        self.leg_mode = None

        self.person_index = -1
        self.activity_index = -1

    def startElement(self, name, attributes):
        self.progress.update()

        if name == "person":
            self.person_id = attributes['id']
            self.ids.append(self.person_id)
            self.person_index += 1

        if name == "plan" and attributes["selected"] == "yes" and not ("freight_" in self.person_id or "cb_" in self.person_id):
            self.plan_selected = True
            self.persons.append(len(self.activities))

        if name =="act" and self.plan_selected:
            self.activity_index += 1
            self.activities.append((
                self.leg_mode, attributes['type'], attributes['facility'],
                convert_time(attributes['end_time']) if 'end_time' in attributes else -1.0,
                convert_time(attributes['start_time']) if 'start_time' in attributes else -1.0
                ))

        if name == "leg" and self.plan_selected:
            self.leg_mode = attributes['mode']

    def endElement(self, name):
        if name == "plan": self.plan_selected = False

    def read(self, path, facility_id_to_index):
        cache = None

        if self.config["use_population_cache"]:
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

        activity_types = []
        activity_modes = []
        activity_facilities = []
        activity_end_times = []
        activity_start_times = []

        for activity in self.activities:
            activity_modes.append(constant.MODES_TO_INDEX[activity[0]] if activity[0] is not None else -1)
            activity_types.append(constant.ACTIVITY_TYPES_TO_INDEX[activity[1]])
            activity_facilities.append(facility_id_to_index[activity[2]])
            activity_end_times.append(activity[3])
            activity_start_times.append(activity[4])

        activity_types = np.array(activity_types, np.int)
        activity_modes = np.array(activity_modes, np.int)
        activity_facilities = np.array(activity_facilities, np.int)
        activity_end_times = np.array(activity_end_times, np.float)
        activity_start_times = np.array(activity_start_times, np.float)

        return activity_types, activity_modes, activity_facilities, activity_end_times, activity_start_times

class PopulationWriter:
    def __init__(self, config):
        self.config = config

    def write(self, input_path, output_path, activity_facilities, facility_ids):
        progress = tqdm(total = len(activity_facilities), desc = "Writing population")
        consume_activities = False
        activity_index = 0

        with utils.open_gzip(output_path, "w+") as fout:
            with utils.open_gzip(input_path, "r") as fin:
                for line in fin:
                    if b"</person" in line:
                        consume_activities = False

                    if b"<person" in line:
                        person_id = re.search(rb'id="(.*?)"', line).group(1)

                        if not (b"freight_" in person_id or b"cb_" in person_id):
                            consume_activities = True

                    if b"<act" in line and consume_activities:
                        line = re.sub(rb'facility="(.*?)"', b'facility="%s"' % bytes(facility_ids[activity_facilities[activity_index]], "ascii"), line)
                        activity_index += 1
                        progress.update()

                    fout.write(line)
        progress.close()
