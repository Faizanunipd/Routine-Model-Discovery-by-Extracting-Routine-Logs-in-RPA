#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import random
from datetime import timedelta
import pandas as pd
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
#from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.obj import EventLog, Trace, Event
from copy import deepcopy

import warnings
warnings.filterwarnings('ignore')



def add_noise_with_probability(ui_log, noise_probability):
    """
    Adds noise to a single UI log segment with a specified probability for each event.
    Noise is introduced by either removing an event or adding a new event.
    """

    noisy_log = []
    # Collect all unique event templates for generating random new events
    event_templates = [event.copy() for event in ui_log]

    i = 0
    while i < len(ui_log):
        if random.random() < noise_probability:
            if random.choice([True, False]):  # Remove the event
                i += 1  # Skip this event
            else:  # Add a new random event
                template_event = random.choice(event_templates).copy()

                # Adjust timestamp if present to maintain order
                if 'time:timestamp' in template_event:
                    prev_time = noisy_log[-1]['time:timestamp'] if noisy_log else template_event['time:timestamp']
                    next_time = ui_log[i]['time:timestamp'] if i < len(ui_log) else template_event['time:timestamp']
                    template_event['time:timestamp'] = prev_time + (next_time - prev_time) / 2

                noisy_log.append(template_event)
        else:
            noisy_log.append(ui_log[i].copy())
            i += 1

    return noisy_log


if __name__ == "__main__":

    for log_name in range(1,10):
        for iter in range(1, 11):
            noise_prob = 0.4
            output_log_dir = f"log{log_name}"

            input_log = f"transform_log{log_name}.xes"
            input_folder = 'Transformed_Logs_and_Results/Our/Transformed_Log_Without_Noise'
            input_path = os.path.join(input_folder, input_log)

            output_folder = f"Transformed_Logs_and_Results/Our/Transformed_Log_With_Noise_{noise_prob}/{output_log_dir}"
            output_log_xes = f"noisy_transform_log{log_name}_{iter}.xes"
            output_log_csv = f"noisy_transform_log{log_name}_{iter}.csv"
            output_path_xes = os.path.join(output_folder, output_log_xes)
            output_path_csv = os.path.join(output_folder, output_log_csv)

            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            
            input_log = xes_importer.apply(input_path)

            df = pm4py.convert_to_dataframe(input_log)
            df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], errors='coerce')
            df['time:timestamp'].ffill(inplace=True)
            input_log = pm4py.convert_to_event_log(df)

            # Add noise with a given ratio
            noisy_log = add_noise_with_probability(input_log, noise_probability=noise_prob)

            # Export the noisy log
            xes_exporter.apply(noisy_log, output_path_xes)

            noisy_log = pm4py.convert_to_dataframe(noisy_log)
            noisy_log.to_csv(output_path_csv, index=False)

        print(f"Add Noie with {noise_prob} in {output_log_dir} successfuly! \n")

# %%
