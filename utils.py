import xml.sax
import pickle
import constant
import os, gzip

def make_xml_parser(handler, f):
    parser = xml.sax.make_parser()
    parser.setContentHandler(handler)
    parser.setFeature(xml.sax.handler.feature_validation, False)
    parser.setFeature(xml.sax.handler.feature_external_ges, False)
    parser.parse(f)

def open_gzip(path, mode = 'r'):
    return gzip.open(path, mode) if path[-3:] == ".gz" else open(path, mode)

def load_cache(name, config):
    path = "%s/%s.p" % (config["cache_path"], name)

    if os.path.isfile(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    else:
        return None

def save_cache(name, data, config):
    path = "%s/%s.p" % (config["cache_path"], name)

    with open(path, "wb+") as f:
        pickle.dump(data, f)
