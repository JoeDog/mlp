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
    self._minimum = minimum
    self._maximum = maximum
    self.notify()

  def register(self, observer):
    self._observers.add(observer)

  def unregister(self, observer):
    self._observers.remove(observer)

  def notify(self):
    for observer in self._observers:
      observer.recalibrate()


class Value(object):
  value     = 0.00
  original  = 0.00
  bounds    = None

  def __init__(self, value, bounds):
    self.original = value
    self.bounds   = bounds
    if (type(value) == bool):
      if self.bounds.min != 0 and self.bounds.max != 1:
        self.bounds.update(0, 1)
      self.value = self.normalize(1.0 if (value==True) else 0.0)
    else:  
      if value <= self.bounds.min:
        self.bounds.update(0 if value >= 0 else value-100, self.bounds.max)
      if value >= self.bounds.max:
        self.bounds.update(self.bounds.min, value+100)
      self.value = self.normalize(value);

  def recalibrate(self)->None: 
    """
    Re-normalizes the value based on a new rangle between minimum and maximum

    Parameters:
    None 
    
    Returns:
    None               Setter method. Resets, minimum, maximum and value
    """
    self.value = normalize(this.original);

  def normalize(self, value:float)->float: 
    return ((value - self.bounds.min) / (self.bounds.max - self.bounds.min))

  def denormalize(self)->float:
    """
    Returns the value in its normal state. If the original value was a boolean, then it will return a bool

    Parameters:
    None

    Returns:
    result(float or bool) 
    """
    if type(self.original) == bool:
      return self.original
    return (self.value * (self.bounds.max - self.bounds.min) + self.bounds.min)

  def toString(self)->str:
    return "Value: min={}, max={}, value={} ({})".format(self.bounds.min, self.bounds.max, self.value, self.original)

class Dataset(object):
  def __init__(self, minimum, maximum):
    self._data   = []
    self.bounds = Range(minimum, maximum)

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

class DataBuilder(object):

  @staticmethod
  def asArray(lo, hi, multiplier=10):
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

