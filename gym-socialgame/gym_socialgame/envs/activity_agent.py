import pandas as pd
import numpy as np
from enum import Enum
from collections.abc import Sequence
from typing import Union

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
		- activity_foresights	| 	Max amount forwards that activity can be moved by price difference
		- activity_values       |   Map from activitiy to Series of active values by time
		- activity_thresholds    |   Map from activity to Series of threshold to consume that activity by time
		- consumed_activities   |   Map from activity to time consumed
		- demand_unit_price_factor | Map from demand unit to Series of energy price factor by time <-- willingness to change usage because of price
									? dependent variables: 1) given time of consumption
									? store in demand unit or activity
		- demand_unit_quantity_factor | Map from demand unit to Series of total energy consumed factor by time <-- willingness to change usage because of price
									? dependent variables: 1) given time of consumption
									? store in demand unit or activity
	'''

	def __init__(self, activity_foresights, activity_values, activity_thresholds, consumed_activities, demand_unit_price_factor, demand_unit_quantity_factor):
		self._activity_foresights = activity_foresights
		self._activity_values = activity_values
		self._activity_thresholds = activity_thresholds
		self._consumed_activities = consumed_activities
		self._demand_unit_price_factor = demand_unit_price_factor
		self._demand_unit_quantity_factor = demand_unit_quantity_factor

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
		- consumed
		- effect_vectors | a map of ActivityConsumers to a map of Activities to a series of effect values/functions affecting activity values by relative time
							? can also condition on the time consumed as effect may differ
	'''

	def __init__(self, demand_units, consumed, effect_vectors):
		self._demand_units = demand_units
		self._consumed = consumed
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

