try:
  from ._version import version as __version__
except ImportError:
  __version__ = 'unknown'

class Range(object):
  def __init__(self, minimum=0, maximum=1):
    self._minimum   = minimum
    self._maximum   = maximum
    self._observers = set()

  @property
  def min(self):
    return self._minimum

  @property
  def max(self):
    return self._maximum

  def update(self, minimum, maximum):
    tmpmin = self._minimum
    tmpmax = self._maximum

    self._minimum = minimum
    self._maximum = maximum

    if tmpmin != self._minimum or tmpmax != self._maximum: 
      # We notify observers only if the values changed
      self.notify()

  def register(self, observer):
    self._observers.add(observer)

  def unregister(self, observer):
    self._observers.remove(observer)

  def notify(self):
    for observer in self._observers:
      observer.recalibrate()


class Value(object):
  _value     = 0.00
  _original  = 0.00
  _bounds    = None
  _asbool    = False

  def __init__(self, value, bounds=Range(0, 1000), analog=False):
    self._original = value
    self._bounds   = bounds
    if (type(value) == bool):
      self._bounds.update(0, 1)
      self._asbool = True
      self._value = self.normalize(1.0 if value==True else 0.0)
    elif analog == True:
      self._bounds.update(0, 1)
      self._value = self.normalize(value)
    else:  
      if value <= self._bounds.min:
        self._bounds.update(0 if value >= 0 else value-100, self._bounds.max)
      if value >= self._bounds.max:
        self._bounds.update(self._bounds.min, value+100)
      self._value = self.normalize(value);

  @property
  def value(self):
    return self._value

  def recalibrate(self)->None: 
    """
    Re-normalizes the value based on a new rangle between minimum and maximum

    Parameters:
    None 
    
    Returns:
    None               Setter method. Resets, minimum, maximum and value
    """
    self._value = normalize(this._original);

  def normalize(self, value:float)->float: 
    return ((value - self._bounds.min) / (self._bounds.max - self._bounds.min))

  def denormalize(self)->float:
    """
    Returns the value in its normal state. If the original value was a boolean, then it will return a bool

    Parameters:
    None

    Returns:
    result(float or bool) 
    """
    if type(self._original) == bool:
      return self.original
    return (self._value * (self.bounds.max - self.bounds.min) + self.bounds.min)

  def toString(self)->str:
    return "Value: min={}, max={}, value={} ({})".format(
      self._bounds.min, self._bounds.max, self.value if self._asbool == False else bool(self._value), self._original
    )

class ValueBuilder(object):

  @staticmethod
  def asArray(lo:int, hi:int, multiplier=10):
    """
    Returns a list of Values from lo to hi. For the purposes of normalization,
    the range is set from lo to hi*multiplier. Adjustments to the multiplier 
    should only be necessary if you run up against the size of an int.

    DataBuilder.asArray(0, 1000) returns a list of Value objects from 0 to 1000

    Parameters:
    lo (int)          The first value in the series
    hi (int)          The last value of the series
    multiplier(int)   The range adjustment for value normalization (default 10)

    Returns:
    list (Value)      A list of Value objects
    """
    r      = lo+hi
    data   = []
    bounds = Range(lo, hi*multiplier)
   
    for i in range(r):
      data.append(Value(i, bounds))

    return data 

  @staticmethod
  def asAnalog(value:float)->Value:
    return Value(value, analog=True)
  
  @staticmethod
  def asBoolean(value:bool)->Value:
    """
    Returns a Value object with a True (1) or False (0) normalized value

    Parameters:
    value(bool):      True or False, the value we want to store

    Returns:
    Value(object) 
    """
    return Value(value)

class Dataset(object):
  def __init__(self, minimum, maximum, multiplier=10):
    self._data   = []
    self.bounds = Range(minimum, maximum*multiplier)

  def add(self, value):
    self._data.append(Value(value, self.bounds))

  def get(self, index):
    return self._data[i]

  def iterator(self):
    class DatasetIterator:
      def __init__(self, array):
        self._data  = array
        self._index = 0

      def __iter__(self):
        return self

      def __next__(self):
        if self._index < len(self._data):
          result = self._data[self._index]
          self._index += 1
          return result
        else:
          raise StopIteration

    return DatasetIterator(self._data)

class Example(object):
  inputs = []
  target = []

  def __init__(self, inputs:list, target:list):
    self.inputs = inputs
    self.target = target

  def getInputs(self)->list:
    return self.inputs

  def getTarget(self)->list:
    return self.target

