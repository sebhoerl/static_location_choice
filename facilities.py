import constant, utils
from tqdm import tqdm
import numpy as np
import xml.sax

class FacilityReader(xml.sax.ContentHandler):
    ID, X, Y, ACTIVITY_TYPES, CAPACITIES = 0, 1, 2, 3, 4

    def __init__(self, config):
        self.facilities = []
        self.progress = None
        self.config = config

        self.activity_type = None

    def startElement(self, name, attributes):
        self.progress.update()

        if name == "facility":
            self.facilities.append([
                attributes['id'],
                attributes['x'],
                attributes['y'],
                set(),
                {}
            ])

        if name == "activity":
            self.facilities[-1][FacilityReader.ACTIVITY_TYPES].add(attributes['type'])
            self.facility_type = attributes['type']

        if name == "capacity":
            self.facilities[-1][FacilityReader.CAPACITIES][self.facility_type] = attributes['value']

    def read(self, path):
        cache = utils.load_cache("facilities", self.config)

        if cache is None:
            self.progress = tqdm(desc = "Loading Facilities ...")
            utils.make_xml_parser(self, utils.open_gzip(path))

            cache = self.process()
            utils.save_cache("facilities", cache, self.config)
        else:
            print("Loaded faciltiies from cache.")

        return cache

    def process(self):
        ids, coordinates = [], []
        capacities = [[] for a in constant.ACTIVITY_TYPES]

        for facility in self.facilities:
            ids.append(facility[FacilityReader.ID])
            coordinates.append((facility[FacilityReader.X], facility[FacilityReader.Y]))

            activity_types = facility[FacilityReader.CAPACITIES]
            for i, a in enumerate(constant.ACTIVITY_TYPES):
                capacities[i].append(activity_types[a] if a in activity_types else 0)

        coordinates = np.array(coordinates, dtype = np.float)
        capacities = np.array(capacities, dtype = np.float).astype(np.int)

        return ids, coordinates, capacities
