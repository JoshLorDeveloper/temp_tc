import pandas as pd
import numpy as np
import json
from enum import Enum
from typing import Union, Callable

from pandas.core.arrays import boolean

class Utilities:
	def special_min(*args):
		return min(i for i in args if i is not None)
		
	def series_add(base : np.ndarray, addition : np.ndarray, addition_displacement = 0):
		for relative_time_step_index, addition_value in enumerate(addition):
			absolute_time_step_index = relative_time_step_index + addition_displacement
			if absolute_time_step_index >= len(base):
				return
			else:
				base[absolute_time_step_index] += addition_value

class ArrayRange:
	def __init__(self, start_index = 0, length = 1):
		self._start_index = start_index
		self._length = length

	def start_index(self):
		return self._start_index
	
	def end_index(self):
		return self.start_index() + len(self)

	def __len__(self):
		return self._length

	def __contains__(self, index):
		return index >= self.start_index() and index < self.end_index()

class ActivityEnvironment:
	'''
	props:
		- activities
		- activitiy_consumers
		- consumed_activities   |   Dict from activity to time consumed
	'''
	def __init__(self, activities, activity_consumers, time_domain):
		self._activities = activities
		self._activity_consumers = activity_consumers
		self._time_domain = time_domain
  
	def execute(self, energy_prices: np.ndarray):
		energy_prices = energy_prices - np.median(energy_prices)
		for time_step_index in range(len(energy_prices)):
			for activity_consumer in self._activity_consumers:
				activity_consumer.execute_step(energy_prices, time_step_index)
	
	def aggregate_demand(self, time_range: ArrayRange = None):
		if time_range is None:
			time_range = self._time_domain
		demand_by_activity_consumer = {}
		for activity_consumer in self._activity_consumers:
			consumer_demand = activity_consumer.aggregate_demand(time_range)
			demand_by_activity_consumer[activity_consumer] = consumer_demand
		return demand_by_activity_consumer

	def build(source_file_name = None):
		if source_file_name is None:
			source_file_name = "gym-socialgame/gym_socialgame/envs/activity_env.json"
		return JsonActivityEnvironmentGenerator.generate_environment(source_file_name)
	
	def restore(self):
		for activity in self._activities:
			activity.restore()
		for activity_consumer in self._activity_consumers:
			activity_consumer.restore()

	def execute_aggregate(self, energy_prices, time_range: ArrayRange = None):
		if time_range is None:
			time_range = self._time_domain
		self.execute(energy_prices)
		result = self.aggregate_demand(time_range)
		return result

	def restore_execute_aggregate(self, energy_prices, time_range: ArrayRange = None):
		if time_range is None:
			time_range = self._time_domain
		self.restore()
		result = self.execute_aggregate(energy_prices, time_range)
		return result

	def build_execute_aggregate(energy_prices, source_file_name = "gym-socialgame/gym_socialgame/envs/activity_env.json"):
		new_env : ActivityEnvironment = ActivityEnvironment.build(source_file_name)
		time_range = new_env._time_domain
		result = new_env.aggregate_execute(energy_prices, time_range)
		return result

	def get_activity_consumers(self):
		return self._activity_consumers


class ActivityConsumer:
	'''
	props:
		- ?activity_foresights	| 	Max amount forwards that activity can be moved by price difference
		- activity_values       |   Dict from activitiy to Series of active values by time
		- activity_thresholds    |   Dict from activity to Series of threshold to consume that activity by time
		- demand_unit_price_factor | Dict from demand unit to Series of energy price factor by time <-- willingness to change usage because of price
									? dependent variables: 1) given time of consumption
									? store in demand unit or activity
		- demand_unit_quantity_factor | Dict from demand unit to Series of total energy consumed factor by time <-- willingness to change usage because of price
									? dependent variables: 1) given time of consumption
									? store in demand unit or activity
		- consumed_activities   |   Dict from activity to time consumed
	'''

	def __init__(self, name, activity_values = None, activity_thresholds = None, demand_unit_price_factor = None, demand_unit_quantity_factor = None):
		
		self.name = name

		if activity_values is None:
			if not hasattr(self, "_activity_values"):
				self._activity_values = {}
		else:
			self._activity_values = activity_values
		
		self._initial_activity_values = ActivityConsumer.copy_activity_values(self._activity_values)
		
		if activity_thresholds is None:
			if not hasattr(self, "_activity_thresholds"):
				self._activity_thresholds = {}
		else:
			self._activity_thresholds = activity_thresholds
		
		if demand_unit_price_factor is None:
			if not hasattr(self, "_demand_unit_price_factor"):
				self._demand_unit_price_factor = {}
		else:
			self._demand_unit_price_factor = demand_unit_price_factor
		
		if demand_unit_quantity_factor is None:
			if not hasattr(self, "_demand_unit_quantity_factor"):
				self._demand_unit_quantity_factor = {}
		else:
			self._demand_unit_quantity_factor = demand_unit_quantity_factor

		self._consumed_activities = {}

	def setup(self, activity_values = None, activity_thresholds = None, demand_unit_price_factor = None, demand_unit_quantity_factor = None):
		
		if activity_values is None:
			if not hasattr(self, "_activity_values"):
				self._activity_values = {}
		else:
			self._activity_values = activity_values

		self._initial_activity_values = ActivityConsumer.copy_activity_values(self._activity_values)

		if activity_thresholds is None:
			if not hasattr(self, "_activity_thresholds"):
				self._activity_thresholds = {}
		else:
			self._activity_thresholds = activity_thresholds
		
		if demand_unit_price_factor is None:
			if not hasattr(self, "_demand_unit_price_factor"):
				self._demand_unit_price_factor = {}
		else:
			self._demand_unit_price_factor = demand_unit_price_factor
		
		if demand_unit_quantity_factor is None:
			if not hasattr(self, "_demand_unit_quantity_factor"):
				self._demand_unit_quantity_factor = {}
		else:
			self._demand_unit_quantity_factor = demand_unit_quantity_factor

	def execute_step(self, energy_price_by_time, time_step):
		to_calculate_for_times = ArrayRange(start_index = time_step)
		for activity, active_value_by_time in self._activity_values.items():
			if (not activity._consumed or activity._consumed is None):
				price_effect_at_time = activity.price_effect_by_time(energy_price_by_time, to_calculate_for_times, self)
				# normalized_price_effect = (price_effect[time_step] - price_effect.median())
				total_value_for_time = active_value_by_time[time_step] + price_effect_at_time
				threshold_for_time = self._activity_thresholds[activity][time_step]

				if total_value_for_time > threshold_for_time:
					self.consume(time_step, activity)

	def consume(self, time_step, activity):
		# print("consumed", time_step, activity.name)
		self._consumed_activities[activity] = time_step
		activity.consume(time_step)

	def aggregate_demand(self, time_range: ArrayRange):
		total_demand = np.zeros(len(time_range), dtype = np.float64)
		for activity, time_consumed in self._consumed_activities.items():
			activity_demand = activity.aggregate_demand(time_consumed, time_range, self)
			Utilities.series_add(total_demand, activity_demand, time_consumed)
			# total_demand = total_demand.add(activity_demand, fill_value=0)
		return total_demand

	def restore(self):
		self._consumed_activities = {}
		self._activity_values = ActivityConsumer.copy_activity_values(self._initial_activity_values)

	def copy_activity_values(activity_values):
		new_dict = {}
		for activity, time_values in activity_values.items():
			new_dict[activity] = time_values.copy()
		return new_dict


class Activity:
	'''
	props:
		- demand_units
		- effect_vectors | a dict of ActivityConsumers to a dict of Activities to a series of effect values/functions affecting activity values by relative time
							? can also condition on the time consumed as effect may differ
		- consumed
	'''
	
	def __init__(self, name, demand_units = None, effect_vectors = None):
		self.name = name
		if demand_units is None:
			if not hasattr(self, "_demand_units"):
				self._demand_units = []
		else:
			self._demand_units = demand_units

		if effect_vectors is None:
			if not hasattr(self, "_effect_vectors"):
				self._effect_vectors = {}
		else:
			self._effect_vectors = effect_vectors
		self._consumed = False

	def setup(self, demand_units = None, effect_vectors = None):
		if demand_units is None:
			if not hasattr(self, "_demand_units"):
				self._demand_units = []
		else:
			self._demand_units = demand_units
			
		if effect_vectors is None:
			if not hasattr(self, "_effect_vectors"):
				self._effect_vectors = {}
		else:
			self._effect_vectors = effect_vectors

	def price_effect_by_time(self, energy_price_by_time, time_range: ArrayRange, for_consumer: ActivityConsumer) -> pd.Series:
		if len(time_range) == 1:
			total = 0
			for demand_unit in self._demand_units:
				# adds price effect by time to total_price_effect
				total += demand_unit.price_effect_by_time(energy_price_by_time, time_range, for_consumer)
			return total
		else:
			total_price_effect = np.zeros(len(time_range), dtype = np.float64)
			for demand_unit in self._demand_units:
				# adds price effect by time to total_price_effect
				demand_unit.price_effect_by_time(energy_price_by_time, time_range, for_consumer, total_price_effect)
			return total_price_effect
	
	def consume(self, time_step):
		self._consumed = time_step
		for for_consumer, effect_vectors_by_activity in self._effect_vectors.items():
			for activity, effect_vector in effect_vectors_by_activity.items():
				active_values = for_consumer._activity_values[activity]
				
				Activity.effect_active_values(time_step, active_values, effect_vector)

	def effect_active_values(base_time_step_index, active_values, local_effect_vector):
		for relative_time_step_index, effect_value in enumerate(local_effect_vector):
			if effect_value == 1:
				continue
			absolute_time_step_index = base_time_step_index + relative_time_step_index
			if absolute_time_step_index >= len(active_values):
				return
			else:
				active_values[absolute_time_step_index] *= effect_value
		

	def aggregate_demand(self, time_consumed, time_range: ArrayRange, for_consumer: ActivityConsumer):
		total_demand = np.zeros(len(time_range), dtype = np.float64)
		for demand_unit in self._demand_units:
			demand_unit_total_demand = demand_unit.absolute_power_consumption_array(time_consumed, for_consumer)
			Utilities.series_add(total_demand, demand_unit_total_demand, time_consumed)
			
		return total_demand
	
	def restore(self):
		self._consumed = False

class DemandUnit:
	'''
	props:
		- power_consumption_array | numpy array representing sequence of consumption
		- ? some way to differentiate quantitative differences between demand units
			qualitative differencesare can be handled by different time units and
			activites
	'''

	def __init__(self, power_consumption_by_time: np.ndarray):
		self._power_consumption_by_time = power_consumption_by_time

	def price_effect_by_time(self, energy_price_by_time, time_range: ArrayRange, for_consumer: ActivityConsumer, for_return:np.ndarray = None) -> pd.Series:

		consumer_price_factor = for_consumer._demand_unit_price_factor[self]
		consumer_quantity_factor = for_consumer._demand_unit_quantity_factor[self]
		
		if for_return is None and len(time_range) != 1:
			for_return = np.zeros(len(time_range), dtype = np.float64)

		for start_time_step_index_delta in range(len(time_range)):
			anchour_time_step_index = time_range.start_index() + start_time_step_index_delta
			total = 0

			for time_step_index_delta, power_consumption in enumerate(self._power_consumption_by_time):
				time_step_index = anchour_time_step_index + time_step_index_delta
				if time_step_index in time_range:
					power_consumed = power_consumption / consumer_quantity_factor[time_step_index]
					effect = power_consumed * energy_price_by_time[time_step_index] * consumer_price_factor[time_step_index]
					total = total + effect
			
			if len(time_range) == 1:
				return total
			for_return[start_time_step_index_delta] += total
		
		return for_return
	
	def absolute_power_consumption_array(self, start_time_step_index, for_consumer: ActivityConsumer):
		consumer_quantity_factor = for_consumer._demand_unit_quantity_factor[self]
		power_consumed_by_time = []
		for time_step_index_delta, power_consumption in enumerate(self._power_consumption_by_time):
			time_step_index = start_time_step_index + time_step_index_delta
			if time_step_index in consumer_quantity_factor:
				power_consumed = power_consumption / consumer_quantity_factor[time_step_index]
				power_consumed_by_time.append(power_consumed)

		return np.array(power_consumed_by_time)

def remove_key_return(original, key, default = None):
	shallow_copy = dict(original)
	val = shallow_copy.pop(key, default)
	return shallow_copy, val

### ENVIRONMENT GENERATOR
class JsonActivityEnvironmentGenerator:	

	def generate_environment(json_file_name):
		with open(json_file_name, "r") as json_file:

			json_data = json.load(json_file)
			
			# initialize time range
			time_range_descriptor = json_data["time_range"]
			length = time_range_descriptor["length"]
			start_index = time_range_descriptor["length"]
			time_range = ArrayRange(start_index = start_index, length = length)

			# initialize named demand units
			named_demand_units = {}

			named_demand_units_data = json_data["named_demand_units"]
			for demand_unit_name, demand_unit_data in named_demand_units_data.items():
				demand_unit_data_array = np.array(demand_unit_data)
				new_demand_unit = DemandUnit(demand_unit_data_array)
				named_demand_units[demand_unit_name] = new_demand_unit

			# initialize activities
			named_activities = {}

			named_activities_data = json_data["activities"]
			for activity_name, activity_data in named_activities_data.items():
				if activity_name != "*":
					new_activity = Activity(activity_name)
					named_activities[activity_name] = new_activity

			activity_list = list(named_activities.values())

			# initialize activity consumers
			named_activity_consumers = {}

			named_activity_consumers_data = json_data["activity_consumers"]
			for activity_consumer_name, activity_consumer_data in named_activity_consumers_data.items():
				if activity_consumer_name != "*":
					new_activity_consumer = ActivityConsumer(activity_consumer_name)
					named_activity_consumers[activity_consumer_name] = new_activity_consumer
			
			activity_consumer_list = list(named_activity_consumers.values())

			# finalize setup of activities
			def activity_property_initilization(activity_data, activity, old_activity_data = None):
				# define demand units
				activity_demand_units = []

				activity_demand_units_data = activity_data["demand_units"]
				for elem in activity_demand_units_data:
					if isinstance(elem, list):
						demand_unit_data_array = np.array(elem)
						demand_unit = DemandUnit(demand_unit_data_array)
						# Adding to named demand units | may want to remove this feature but would reqcuire removing unnamed demand units
						named_demand_units[demand_unit] = demand_unit
					elif isinstance(elem, str):
						demand_unit = named_demand_units[elem]

					activity_demand_units.append(demand_unit)

				# define effect vectors
				activity_effect_vectors = {}

				### create actvity vectors once we know the activity consumers
				activity_effect_vectors_data = activity_data["effect_vectors"]

				### setup functions to run through json data
				generalize_effect_vector_over_times_function = JsonActivityEnvironmentGenerator.array_over_value_function(
																			time_range
																		)

				generalize_effect_vector_over_activities_function = JsonActivityEnvironmentGenerator.dict_over_value_function(
																			named_activities,
																			generalize_effect_vector_over_times_function
																		)

				generalize_effect_vector_over_consumers_function = JsonActivityEnvironmentGenerator.dict_over_value_function(
																			named_activity_consumers, 
																			generalize_effect_vector_over_activities_function
																		)

				activity_effect_vectors = generalize_effect_vector_over_consumers_function(activity_effect_vectors_data, old_dict_value = activity._effect_vectors)

				# setup activity with found information
				activity.setup(activity_demand_units, activity_effect_vectors)

			JsonActivityEnvironmentGenerator.loop_over_value(named_activities_data, named_activities, activity_property_initilization)

			# finalize setup of activity consumers			
			def activity_consumer_property_initilization(activity_consumer_data, activity_consumer, old_activity_consumer_data = None):
				
				actvity_consumer_setup_args = {}

				# define activity values
				if "activity_values" in activity_consumer_data:
					activity_values_data = activity_consumer_data["activity_values"]

					#### setup functions to run through json data
					generalize_consumer_value_over_times_function = JsonActivityEnvironmentGenerator.array_over_value_function(
																				time_range
																			)

					generalize_consumer_value_over_activities_function = JsonActivityEnvironmentGenerator.dict_over_value_function(
																				named_activities,
																				generalize_consumer_value_over_times_function
																			)
					
					### create property using defined functions
				
					actvity_consumer_setup_args["activity_values"] = generalize_consumer_value_over_activities_function(activity_values_data, old_dict_value = activity_consumer._activity_values)

				# define activity thresholds
				if "activity_thresholds" in activity_consumer_data:
					activity_thresholds_data = activity_consumer_data["activity_thresholds"]

					### setup functions to run through json data | First function unnecessary at the moment
					generalize_consumer_value_over_times_function = JsonActivityEnvironmentGenerator.array_over_value_function(
																				time_range
																			)

					generalize_consumer_value_over_activities_function = JsonActivityEnvironmentGenerator.dict_over_value_function(
																				named_activities,
																				generalize_consumer_value_over_times_function
																			)
					### create property using defined functions
					actvity_consumer_setup_args["activity_thresholds"] = generalize_consumer_value_over_activities_function(activity_thresholds_data, old_dict_value = activity_consumer._activity_thresholds)

				# define demand unit price factors
				if "demand_unit_price_factors" in activity_consumer_data:
					demand_unit_price_factors_data = activity_consumer_data["demand_unit_price_factors"]

					### setup functions to run through json data | First function unnecessary at the moment
					generalize_consumer_value_over_times_function = JsonActivityEnvironmentGenerator.array_over_value_function(
																				time_range
																			)

					generalize_consumer_value_over_demand_units_function = JsonActivityEnvironmentGenerator.dict_over_value_function(
																				named_demand_units,
																				generalize_consumer_value_over_times_function
																			)
					### create property using defined functions
					actvity_consumer_setup_args["demand_unit_price_factor"] = generalize_consumer_value_over_demand_units_function(demand_unit_price_factors_data, old_dict_value = activity_consumer._demand_unit_price_factor)

				# define demand unit quantity factors
				if "demand_unit_quantity_factors" in activity_consumer_data:
					demand_unit_quantity_factors_data = activity_consumer_data["demand_unit_quantity_factors"]

					### setup functions to run through json data | First function unnecessary at the moment
					generalize_consumer_value_over_times_function = JsonActivityEnvironmentGenerator.array_over_value_function(
																				time_range
																			)

					generalize_consumer_value_over_demand_units_function = JsonActivityEnvironmentGenerator.dict_over_value_function(
																				named_demand_units,
																				generalize_consumer_value_over_times_function
																			)

					### create property using defined functions
					actvity_consumer_setup_args["demand_unit_quantity_factor"] = generalize_consumer_value_over_demand_units_function(demand_unit_quantity_factors_data, old_dict_value = activity_consumer._demand_unit_quantity_factor)

				# setup activity consumer with found information
				activity_consumer.setup(**actvity_consumer_setup_args)
			
			JsonActivityEnvironmentGenerator.loop_over_value(named_activity_consumers_data, named_activity_consumers, activity_consumer_property_initilization)

			return ActivityEnvironment(activity_list, activity_consumer_list, time_range)

	# loops over values in json data and calls function on them | Effectively a map function
	def loop_over_value(property_json_data, property_name_dict, function_on_child_value = None):

		property_dict = JsonActivityEnvironmentGenerator.over_full_dict_property(property_json_data, property_name_dict, function_on_child_value)
		return list(property_dict.keys())
		

	# activites, demand_units, activity_consumers
	def dict_over_value_function(property_name_dict, function_on_child_value = None):
		def dict_over_value(property_json_data, parent = None, old_dict_value = None):

			property_dict = JsonActivityEnvironmentGenerator.over_full_dict_property(property_json_data, property_name_dict, function_on_child_value, old_dict_value)
			return property_dict

		return dict_over_value

	# time
	def array_over_value_function(array_range: ArrayRange, function_on_child_value = None):
		def array_over_value(property_json_data, parent = None, old_array_value = None):

			property_array = JsonActivityEnvironmentGenerator.over_full_array_property(property_json_data, array_range, function_on_child_value, old_array_value)
			return property_array

		return array_over_value

	# activites, demand_units, activity_consumers
	def over_full_dict_property(object_json_data, named_objects, function_on_value = None, to_return = {}):
		object_list = list(named_objects.values())

		if to_return is None:
			to_return = {} 

		object_json_data, general_object_data = remove_key_return(object_json_data, "*")

		if general_object_data is not None:
			JsonActivityEnvironmentGenerator.generalize_dict(
				general_object_data, 
				object_list, 
				to_return, 
				function_on_value
			)

		for specific_object_name, specific_object_data in object_json_data.items():
			if specific_object_name in named_objects:
				specific_object = named_objects[specific_object_name]

				if function_on_value is not None:
					to_return[specific_object] = function_on_value(specific_object_data, specific_object, to_return.get(specific_object))
				else:
					to_return[specific_object] = specific_object_data
		
		return to_return
	
	# time
	def over_full_array_property(object_json_data, array_range: ArrayRange, function_on_value = None, to_return = None):

		array_values = []
		
		object_json_data, general_object_data = remove_key_return(object_json_data, "*")

		if general_object_data is not None:

			array_values = JsonActivityEnvironmentGenerator.generalize_list(
				general_object_data, 
				array_range, 
				function_on_value
			)
			
			to_return = np.array(array_values)
		elif to_return is None:

			array_values = JsonActivityEnvironmentGenerator.generalize_list(
				None, 
				array_range
			)
			
			to_return = np.array(array_values)

		for series_key, specific_object_data in object_json_data.items():
			int_key = int(series_key)
			int_index = int_key - array_range.start_index()
			if int_index in array_range:
				if function_on_value is not None:
					to_return[int_key] = function_on_value(specific_object_data, series_key, to_return.get(int_key))
				else:
					to_return[int_key] = specific_object_data
		
		return to_return

	def generalize_dict(value_to_generalize, generalize_over, to_return = {}, function_on_value = None):
		for key in generalize_over:
			if function_on_value is not None:
				to_return[key] = function_on_value(value_to_generalize, key, to_return.get(key))
			else:
				to_return[key] = value_to_generalize
		return to_return

	def generalize_list(value_to_generalize, array_range: ArrayRange, function_on_value = None):
		series_values = []
		for index in range(array_range.start_index(), array_range.end_index()):
			if function_on_value is not None:
				series_values.append(function_on_value(value_to_generalize, index))
			else:
				series_values.append(value_to_generalize)
		return series_values


val = JsonActivityEnvironmentGenerator.generate_environment("gym-socialgame/gym_socialgame/envs/activity_env.json")
times = val._time_domain
energy_prices = np.random.uniform(low=20, high=50, size=(len(times),))
energy_prices_by_time = energy_prices
for i in range(1, 100):
	val.restore_execute_aggregate(energy_prices_by_time)
# val.execute(energy_prices_by_time)
# result = val.compile_demand()
# for consumer in result:
# 	print("For Actvity Consumer ", consumer)
# 	with pd.option_context('display.max_rows', None, 'display.max_columns', None):
# 		print(result[consumer])

# print(val)
# print("Activities")
# for activity in val._activities:
# 	demand_units = []
# 	for demand_unit in activity._demand_units:
# 		demand_units.append(demand_unit._power_consumption_array)
# 	print("Activity: ", demand_units)

# print("Activity Consumers")
# for activity_consumer in val._activity_consumers:
# 	print("Activity Consumer: ", activity_consumer._activity_thresholds)
		

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
	def execute(value : Union[float, SelfDescribedValue], action: Callable, *args, **kwargs):
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

