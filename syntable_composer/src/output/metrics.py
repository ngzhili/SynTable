import os
import yaml


class Metrics:
    """ For managing performance metrics of dataset generation. """

    def __init__(self, log_dir, content_log_path):
        """ Construct Metrics. """

        self.metric_path = os.path.join(log_dir, "metrics.txt")
        self.content_log_path = content_log_path

    def output_performance_metrics(self):
        """ Collect per-scene metrics and calculate and output summary metrics. """

        with open(self.content_log_path, "r") as f:
            log = yaml.safe_load(f)

        durations = []
        for log_entry in log:
            if type(log_entry["index"]) is int:
                durations.append(log_entry["time_elapsed"])
        durations.sort()

        metric_packet = {}

        n = len(durations)
        metric_packet["time_per_sample_min"] = durations[0]
        metric_packet["time_per_sample_first_quartile"] = durations[n // 4]
        metric_packet["time_per_sample_median"] = durations[n // 2]
        metric_packet["time_per_sample_third_quartile"] = durations[3 * n // 4]
        metric_packet["time_per_sample_max"] = durations[-1]

        metric_packet["time_per_sample_mean"] = sum(durations) / n

        with open(self.metric_path, "w") as f:
            yaml.safe_dump(metric_packet, f)
