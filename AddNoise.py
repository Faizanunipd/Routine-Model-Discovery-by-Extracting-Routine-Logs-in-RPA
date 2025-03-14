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



def add_noise_with_probability(event_log: EventLog, noise_probability: float) -> EventLog:
    """
    Adds noise to an event log with a specified probability for each event.
    Noise is introduced by either removing an event or adding a new event.
    """

    noisy_log = EventLog()
    # Collect all unique event templates for generating random new events
    event_templates = [deepcopy(event) for trace in event_log for event in trace]

    for trace in event_log:
        noisy_trace = Trace(attributes=deepcopy(trace.attributes))  # Copy trace attributes
        
        i = 0  # Index for the event being processed in the original trace
        while i < len(trace):
            if random.random() < noise_probability:  # Decide whether to apply noise
                if random.choice([True, False]):  # Remove the event
                    # Skip this event by not copying it to noisy_trace
                    i += 1  # Move to the next event
                else:  # Add a new random event
                    template_event = random.choice(event_templates)
                    new_event = Event(deepcopy(template_event))
                    
                    # Adjust timestamp to ensure uniqueness
                    if 'time:timestamp' in template_event:
                        prev_time = noisy_trace[-1]['time:timestamp'] if noisy_trace else template_event['time:timestamp']
                        next_time = trace[i]['time:timestamp'] if i < len(trace) else template_event['time:timestamp']
                        new_event['time:timestamp'] = prev_time + (next_time - prev_time) / 2
                    
                    noisy_trace.append(new_event)  # Append the new event
            else:
                # Copy the current event to the noisy trace
                noisy_trace.append(deepcopy(trace[i]))
                i += 1  # Move to the next event
        
        # Append the modified trace to the noisy log
        noisy_log.append(noisy_trace)

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
