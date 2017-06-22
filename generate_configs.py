import json

with open("config.json") as f:
    template = json.load(f)

names = []

for scenario in [1, 10, 100]:
    for times in ["times", "notimes"]:
        for constraint in ["aggressive", "moderate"]:
            template["capacity_likelihood_alpha"] = (1e-4 if constraint == "aggressive" else 1e-5) / scenario
            template["uses_times"] = "times" == times

            template["source_facilities_path"] = "data/lc_prepare/facilities_%d.xml.gz" % scenario
            template["source_population_path"] = "data/lc_prepare/population_%d.xml.gz" % scenario
            template["capacity_scaling_factor"] = scenario / 100.0

            template["use_population_cache"] = False
            template["use_facilities_cache"] = False

            template["output_path"] = "output_%d_%s_%s" % (scenario, times, constraint)
            names.append("%d_%s_%s" % (scenario, times, constraint))

            with open("configs/config_%d_%s_%s.json" % (scenario, times, constraint), "w+") as f:
                json.dump(template, f)

runner = ["# parallel 10"]
for name in names:
    runner = ["mkdir -p output_%s" % name] + runner
    runner.append("python3 run.py configs/config_%s.json" % name)

with open("runner.sh", "w+") as f:
    f.write("\n".join(runner))
