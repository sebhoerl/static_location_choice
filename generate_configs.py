import json

with open("config.json") as f:
    template = json.load(f)

names = []

for scenario in [1, 10, 100]:
    for times in ["notimes"]: #["times", "notimes"]:
            template["capacity_likelihood_alpha"] = 1e-4 / scenario
            template["uses_times"] = "times" == times

            template["source_facilities_path"] = "data/lc_prepare/facilities_%d.xml.gz" % scenario
            template["source_population_path"] = "data/lc_prepare/population_%d.xml.gz" % scenario
            template["capacity_scaling_factor"] = scenario / 100.0

            template["use_population_cache"] = False
            template["use_facilities_cache"] = False

            template["output_path"] = "output_%d_%s" % (scenario, times)
            names.append("%d_%s" % (scenario, times))

            with open("configs/config_%d_%s.json" % (scenario, times), "w+") as f:
                json.dump(template, f)

runner = ["# parallel 10"]
for name in names:
    runner = ["mkdir -p output_%s/dists" % name] + runner
    runner.append("python3 run.py configs/config_%s.json" % name)

with open("runner.sh", "w+") as f:
    f.write("\n".join(runner))
