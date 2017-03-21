config = dict(
    cache_path = "cache",
    use_population_cache = True,
    sampling_factor = 1e12,
    output_interval = int(1e5),
    measurement_interval = int(1e3),
    total_iterations = int(1e6),
    validation_interval = None,
    output_population_interval = None,#int(1e4),
    relevant_modes = ["car", "pt", "bike", "walk"],
    relevant_activity_types = ["shop", "leisure", "escort_kids", "escort_other", "remote_work"],
    additional_activity_types = ["work", "education"],
    source_facilities_path = "data/facilities.xml.gz",
    source_population_path = "data/population.xml.gz",
    target_population_path = "output/population.xml.gz",
    capacity_scaling_factor = 0.01,
    distribution_mode = "random",
    minimum_time = 0.0,
    maximum_time = 24.0 * 3600,
    time_bins = 24 * 12,
    proposal = "advanced",
    capacity_likelihood_alpha = 1e-3,
    distance_based_proposal_candidate_set_size = 30,
    override_sigma = {
        #("walk", "*") : 1e-3
    }
)
