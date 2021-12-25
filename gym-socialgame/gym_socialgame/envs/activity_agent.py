import pandas as pd
import numpy as np
from enum import Enum
from collections.abc import Sequence
from typing import Union
import json

class ActivityEnvironment:
	'''
	props:
		- activities
		- activitiy_consumers
		- consumed_activities   |   Map from activity to time consumed
	'''
	def __init__(self, activities, activity_consumers):
		self._activities = activities
		self._activity_consumers = activity_consumers
  
	def execute(self, for_times: np.ndarray):
		for time_step in for_times:
			for activity_consumer in self._activity_consumers:
				activity_consumer.execute_step(time_step)
	
	def compile_demand(self, for_times: np.ndarray):
		demand_by_activity_consumer = {}
		for activity_consumer in self._activity_consumers:
			consumer_demand = activity_consumer.compile_demand(for_times)
			demand_by_activity_consumer[activity_consumer] = consumer_demand
		return demand_by_activity_consumer
		
class ActivityConsumer:
	'''
	props:
		- ?activity_foresights	| 	Max amount forwards that activity can be moved by price difference
		- activity_values       |   Map from activitiy to Series of active values by time
		- activity_thresholds    |   Map from activity to Series of threshold to consume that activity by time
		- demand_unit_price_factor | Map from demand unit to Series of energy price factor by time <-- willingness to change usage because of price
									? dependent variables: 1) given time of consumption
									? store in demand unit or activity
		- demand_unit_quantity_factor | Map from demand unit to Series of total energy consumed factor by time <-- willingness to change usage because of price
									? dependent variables: 1) given time of consumption
									? store in demand unit or activity
		- consumed_activities   |   Map from activity to time consumed
	'''

	def __init__(self, activity_values = [], activity_thresholds = [], demand_unit_price_factor = [], demand_unit_quantity_factor = []):
		self._activity_values = activity_values
		self._activity_thresholds = activity_thresholds
		self._demand_unit_price_factor = demand_unit_price_factor
		self._demand_unit_quantity_factor = demand_unit_quantity_factor
		self._consumed_activities = []

	def execute_step(self, time_step):
		to_calculate_for_times = np.array([time_step])
		for activity, active_value_by_time in self._activity_values.items():
			if (not activity._consumed or activity._consumed is None):
				price_effect = activity.price_effect_by_time(to_calculate_for_times, self)
				total_value_for_time = (active_value_by_time + price_effect)[time_step]
				threshold_for_time = self._consumed_activities[activity][time_step]
				if total_value_for_time > threshold_for_time:
					self.consume(activity)

	def consume(self, time_step, activity: Activity):
		self._consumed_activities[activity] = time_step
		activity.consume(time_step, self)

	def compile_demand(self, for_times: np.ndarray):
		total_demand = pd.Series(np.full(len(for_times), 0), index=for_times)
		for activity, time_consumed in self._consumed_activities.items():
			actvity_demand = activity.compile_demand(time_consumed, for_times)
			total_demand = total_demand + actvity_demand
		return total_demand


class Activity:
	'''
	props:
		- demand_units
		- effect_vectors | a map of ActivityConsumers to a map of Activities to a series of effect values/functions affecting activity values by relative time
							? can also condition on the time consumed as effect may differ
		- consumed
	'''
	
	def __init__(self, demand_units = [], effect_vectors = []):
		self._demand_units = demand_units
		self._effect_vectors = effect_vectors
		self._consumed = False

	def setup(self, demand_units = [], effect_vectors = []):
		self._demand_units = demand_units
		self._effect_vectors = effect_vectors

	def price_effect_by_time(self, for_times: np.ndarray, for_consumer: ActivityConsumer) -> pd.Series:
		total_price_effect = pd.Series(np.full(len(for_times), 0), index=for_times)
		for demand_unit in self._demand_units:
			price_effect = demand_unit.price_effect_by_time(for_times, for_consumer)
			total_price_effect = total_price_effect + price_effect
		return total_price_effect
	
	def consume(self, time_step, by_consumer: ActivityConsumer):
		self._consumed = time_step
		effect_vectors_by_activity = self._effect_vectors[by_consumer]
		for activity, effect_vector in effect_vectors_by_activity.items():
			# generate effect vector
			local_effect_vector = np.copy(effect_vector, subok=True)
			local_effect_vector.index = local_effect_vector.index + time_step # Add time delta to start time, note: all need to be timestamps and time deltas or all floats
			# change active values of activity by time for activity consumer
			active_values_by_time = by_consumer._activity_values[activity]
			new_active_values_by_time = active_values_by_time * local_effect_vector # change for increased complexity
			by_consumer._activity_values[activity] = new_active_values_by_time

	def compile_demand(self, time_consumed, for_times: np.ndarray):
		total_demand = pd.Series(np.full(len(for_times), 0), index=for_times)
		for demand_unit in self._demand_units:
			demand_unit_total_demand = demand_unit.absolute_power_consumption_array(time_consumed)
			total_demand = total_demand + demand_unit_total_demand
		return total_demand

class DemandUnit:
	'''
	props:
		- power_consumption_array | numpy array representing sequence of consumption
		- ? some way to differentiate quantitative differences between demand units
			qualitative differencesare can be handled by different time units and
			activites
	'''

	def __init__(self, power_consumption_array):
		self._power_consumption_array = power_consumption_array

	def price_effect_by_time(self, for_times: np.ndarray, for_consumer: ActivityConsumer) -> pd.Series:
		consumer_price_factor = for_consumer._demand_unit_price_factor
		consumer_quantity_factor = for_consumer._demand_unit_quantity_factor
		
		price_effects = []
		for start_time_step in for_times:
			total = 0

			for time_step_delta, power_consumption in enumerate(self._power_consumption_array):
				time_step = start_time_step + time_step_delta
				power_consumed = power_consumption / consumer_quantity_factor[time_step]
				effect = power_consumed * consumer_price_factor[time_step]
				total = total + effect
			
			price_effects.append(total)
		
		return pd.Series(price_effects, index=for_times)
	
	def absolute_power_consumption_array(self, start_time_step, for_consumer: ActivityConsumer):
		consumer_quantity_factor = for_consumer._demand_unit_quantity_factor
		power_consumed_by_time = []
		for time_step_delta, power_consumption in enumerate(self._power_consumption_array):
			time_step = start_time_step + time_step_delta
			power_consumed = power_consumption / consumer_quantity_factor[time_step]
			power_consumed_by_time.append(power_consumed)

		absolute_times = self._power_consumption_array.index + start_time_step

		return pd.Series(power_consumed_by_time, index=absolute_times)


### ENVIRONMENT GENERATOR
class JsonActivityEnvironmentGenerator:	

	def generate_environment(json_file_name):
		with open(json_file_name) as json_file:
			json_data = json.load(json_file)
			
			# initialize times
			time_range_descriptor = json_data["times"]
			start = time_range_descriptor["start"]
			stop = time_range_descriptor["stop"]
			interval = time_range_descriptor["interval"]
			times = np.arange(start, stop, interval)

			# initialize named demand units
			named_demand_units = {}

			named_demand_units_data = json_data["named_demand_units"]
			for demand_unit_name, demand_unit_data in named_demand_units_data.items():
				new_demand_unit = DemandUnit(demand_unit_data)
				named_demand_units[demand_unit_name] = new_demand_unit

			# initialize activities
			named_activities = {}

			named_activities_data = json_data["activities"]
			for activity_name, activity_data in named_activities_data.items():

				new_activity = Activity()
				named_activities[activity_name] = new_activity

			activity_list = list(named_activities.values())

			# initialize activity consumers
			named_activity_consumers = {}

			named_activity_consumers_data = json_data["activity_consumers"]
			for activity_consumer_name, activity_consumer_data in named_activity_consumers_data.items():

				new_activity_consumer = ActivityConsumer()
				named_activity_consumers[activity_consumer_name] = new_activity_consumer
			
			activity_consumer_list = list(named_activity_consumers.values())

			# finalize setup of activities
			for activity_name, activity_data in named_activities_data.items():

				# define demand units
				activity_demand_units = []

				activity_demand_units_data = activity_data["demand_units"]
				for elem in activity_demand_units_data:
					if isinstance(elem, list):
						demand_unit = DemandUnit(elem)
					elif isinstance(elem, str):
						demand_unit = named_demand_units[elem]

					activity_demand_units.append(demand_unit)

				# define effect vectors
				activity_effect_vectors = {}

				# create actvity vectors once we know the activity consumers
				activity_effect_vectors_data = activity_data["effect_vectors"]

				# setup functions to run through json data
				generalize_effect_vector_over_times_function = JsonActivityEnvironmentGenerator.generalize_value_over_time_function(
																			times
																		)

				generalize_effect_vector_over_activities_function = JsonActivityEnvironmentGenerator.generalize_value_over_activity_function(
																			named_activities,
																			generalize_effect_vector_over_times_function
																		)

				generalize_effect_vector_over_consumers_function = JsonActivityEnvironmentGenerator.generalize_value_over_consumer_function(
																			named_activity_consumers, 
																			generalize_effect_vector_over_activities_function
																		)

				activity_effect_vectors = generalize_effect_vector_over_consumers_function(activity_effect_vectors_data)

				# setup activity with found information
				named_activities[activity_name].setup(activity_demand_units, activity_effect_vectors)

			##################
			##################

			# initialize activity consumers
			named_activity_consumers = {}

			named_activity_consumers_data = json_data["activity_consumers"]

			# process dynamic
			dynamic_activity_consumer = named_activity_consumers_data.pop('*', None)
			dynamic_activity_values_data = dynamic_activity_consumer["activity_values"]
			JsonActivityEnvironmentGenerator.generalize_map(dynamic_activity_values_data, activity_list)

			for activity_consumer_name, activity_consumer_data in named_activity_consumers_data.items():
				
				activity_values_data = activity_consumer_data["activity_values"]


	def activity_effect_vectors_generation_function(activity_list, times):
		def activity_effect_vectors_function_on_value(value_to_generalize, key = None):
			to_return = {}
			# process dynamic
			dynamic_effect_on_activitiy = value_to_generalize.pop('*', None)
			JsonActivityEnvironmentGenerator.generalize_map(
				dynamic_effect_on_activitiy, 
				activity_list, 
				to_return, 
				JsonActivityEnvironmentGenerator.time_series_from_dict_generation_function(times)
			)
		return activity_effect_vectors_function_on_value	
	
	def generalize_value_over_consumer_function(named_consumers, function_on_child_value = None):
		def generalize_value_over_consumer(consumer_json_data, parent = None):

			consumer_map = JsonActivityEnvironmentGenerator.generalize_over_map_property(consumer_json_data, named_consumers, function_on_child_value)
			return consumer_map

		return generalize_value_over_consumer
	
	def generalize_value_over_activity_function(named_activities, function_on_child_value = None):
		def generalize_value_over_activity(activity_json_data, parent = None):

			activity_map = JsonActivityEnvironmentGenerator.generalize_over_map_property(activity_json_data, named_activities, function_on_child_value)
			return activity_map

		return generalize_value_over_activity
	
	def generalize_value_over_demand_unit_function(named_demand_units, function_on_child_value = None):
		def generalize_value_over_demand_unit(demand_unit_json_data, parent = None):

			demand_unit_map = JsonActivityEnvironmentGenerator.generalize_over_map_property(demand_unit_json_data, named_demand_units, function_on_child_value)
			return demand_unit_map

		return generalize_value_over_demand_unit

	def generalize_value_over_time_function(times, function_on_child_value = None):
		def generalize_value_over_time(time_json_data, parent = None):

			time_series = JsonActivityEnvironmentGenerator.generalize_over_series_property(time_json_data, times, function_on_child_value)
			return time_series

		return generalize_value_over_time

	# activites, demand_units, activity_consumers
	def generalize_over_map_property(object_json_data, named_objects, to_return = {}, function_on_value = None):
		object_list = list(named_objects.values())

		general_object_data = object_json_data.pop('*', None)
		if general_object_data is not None:
			JsonActivityEnvironmentGenerator.generalize_map(
				general_object_data, 
				object_list, 
				to_return, 
				function_on_value
			)

		for specific_object_name, specific_object_data in object_json_data.items():
			if specific_object_name in named_objects:
				specific_object = named_objects[specific_object_name]

				if function_on_value is not None:
					to_return[specific_object] = function_on_value(specific_object_data, specific_object)
				else:
					to_return[specific_object] = specific_object_data
		
		return to_return
	
	# time
	def generalize_over_series_property(object_json_data, series_keys_list, function_on_value = None):

		series_values = []

		general_object_data = object_json_data.pop('*', None)
		if general_object_data is not None:
			series_values = JsonActivityEnvironmentGenerator.generalize_list(
				general_object_data, 
				series_keys_list, 
				function_on_value
			)
		else:
			series_values = JsonActivityEnvironmentGenerator.generalize_list(
				None, 
				series_keys_list
			)

		to_return = pd.Series(series_values, index=series_keys_list)

		for series_key, specific_object_data in object_json_data.items():
			if series_key in series_keys_list:
				if function_on_value is not None:
					to_return[series_key] = function_on_value(specific_object_data, series_key)
				else:
					to_return[series_key] = specific_object_data
		
		return to_return

	def generalize_map(value_to_generalize, generalize_over, to_return = {}, function_on_value = None):
			for key in generalize_over:
				if function_on_value is not None:
					to_return[key] = function_on_value(value_to_generalize, key)
				else:
					to_return[key] = value_to_generalize
			return to_return

	def generalize_list(value_to_generalize, generalize_over, function_on_value = None):
		series_values = []
		for key in generalize_over:
			if function_on_value is not None:
				series_values.append(function_on_value(value_to_generalize, key))
			else:
				series_values.append(value_to_generalize)
		return series_values
	



class ActivityEnvironmentGenerator:

	def __init__(self):
		self.DEMAND_SAMPLE = np.array([ 0.28,  11.9,   16.34,  16.8,  17.43,  16.15,  16.23,  15.88,  15.09,  35.6,
									123.5,  148.7,  158.49, 149.13, 159.32, 157.62, 158.8,  156.49, 147.04,  70.76,
									42.87,  23.13,  22.52,  16.8 ])
		self.DEMAND_UNIT_SIZE = 3
		self.DEMAND_UNIT_SIZE = 3

	def generate_environment(self):
		
		sample_demand_units = self.DEMAND_SAMPLE.reshape(-1, self.DEMAND_UNIT_SIZE)
		demand_units = []
		for sample_demand in sample_demand_units:
			demand_units.append(DemandUnit(sample_demand))
		
		activities = []
		for demand_unit in demand_units:
			effect_vector = []
			activities.append(Activity(demand_unit, effect_vector))
		
		activity_consumers = []




### FUTURE WORK
class SelfDescribedValue:
	'''
	props:
		- condition_on | a single value that this layer of the self described value may be conditioned on
		- value | 0 or 1 dimensional array of SelfDescribedValues
		- type | if condition_on is non-empty = LIST or DICT or SERIES otherwise is either NUMBER or FUNCTION
	'''
	pass


class ValueType(Enum):
	NUMBER = 1
	FUNCTION = 2
	LIST = 3
	DICT = 4
	SERIES = 5


class SelfDescribedValueOperations:
	def execute(value : Union[float, SelfDescribedValue], action: function, *args, **kwargs):
		if (type(value) == float):
			action(value)
		elif (value.type == ValueType.NUMBER):
			action(value.value)
		elif (value.type == ValueType.FUNCTION):
			action(value.value, args, kwargs)
		elif (value.type == ValueType.LIST):
			conditioned_on = value.condition_on
			for index, elem in enumerate(value.value):
				SelfDescribedValueOperations.execute(elem, action, *args, **{conditioned_on: index}, **kwargs)
		elif (value.type == ValueType.DICT):
			assert type(value.value) == dict
			conditioned_on = value.condition_on
			for key, elem in value.value.items():
				SelfDescribedValueOperations.execute(elem, action, *args, **{conditioned_on: key}, **kwargs)
		elif (value.type == ValueType.SERIES):
			assert type(value.value) == pd.Series
			conditioned_on = value.condition_on
			for key, elem in value.value.items():
				SelfDescribedValueOperations.execute(elem, action, *args, **{conditioned_on: key}, **kwargs)

