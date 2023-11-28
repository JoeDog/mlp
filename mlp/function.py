import math
import importlib
from   enum          import IntEnum

try:
  from ._version import version as __version__
except ImportError:
  __version__ = 'unknown'

class Type(IntEnum):
  UNKNOWN    = 0
  GAUSSIAN   = 1
  HEAVISIDE  = 2
  HYPERBOLIC = 3
  LOGISTIC   = 4
  RELU       = 5
  SIGMOIDAL  = 6 
  TAHN       = 7  
 
  @staticmethod
  def toString(self) -> str:
    """
    Returns a string representing the data in this sequence.

    Parameters:
    None

    Returns:
    str               A string representation of this series of characters.
    """
    return self.__str__()
 
  @staticmethod
  def fromString(name):
    """
    Returns an in representing the enum value

    Parameters:
    name (str)          A string representing the enum's name, i.e., 'GAUSSIAN'

    Returns 
    int                 An int value for enum name 
    """
    try:
      return Type[name]
    except KeyError:
      return Type["UNKNOWN"]

  def __str__(self):
    return self.name


class Factory:

  def __init__(self, form=Type.RELU):
    self.form = form
    self.func = {}

    self._create()

  def function(self):
    return self.form

  def evaluate(self, val)->float:
    clzz = self.func[str(Type(self.form))]
    return clzz.evaluate(val) 
  
  def derivate(self, val)->float:
    clzz = self.func[str(Type(self.form))]
    return clzz.derivate(val) 

  def set(self, form):
    self.form = form

  def toString(self)->str:
    clzz = self.func[str(Type(self.form))]
    return clzz
  
  def _create(self)->None:
    for i in range(len(list(Type))):
      name = str(Type(i))
      if i == 0:
        self.func[name] = None
      else:
        module  = importlib.import_module("mlp.function", package='.')
        MyClass = getattr(module, name.capitalize())
        clss = MyClass()
        self.func[name] = clss

class Gaussian:

  def evaluate(self, val:float)->float: 
    d = math.pow(-val, 2);
    f = 1 / (math.exp(d));
    return f;

  def derivate(self, val:float)->float:
    return -2 * val * self.evaluate(val);

  def __str__(self)->str:
    return "Gaussian"

class Heaviside:

  def evaluate(self, val:float)->float: 
    if val >= 0.0:
      return 1.0
    else: 
      return 0.0

  def derivate(self, val:float)->float:
    return 1.0

  def __str__(self)->str:
    return "Heaviside"

class Hyperbolic:

  def evaluate(self, val:float)->float: 
    return math.tanh(val)

  def derivate(self, val:float)->float:
    return (1 - math.pow(self.evaluate(val), 2));

  def __str__(self)->str:
    return "Hyperbolic"

class Logistic:

  def evaluate(self, val:float)->float: 
    return 0.0

  def derivate(self, val:float)->float:
    return 1.0

  def __str__(self)->str:
    return "Logistic"

class Relu:

  def evaluate(self, val:float)->float: 
    return 1 / (1 + math.exp(-val))
    
  def derivate(self, val:float)->float:
    return math.exp(val) / math.pow(1 + math.exp(val), 2);

  def __str__(self)->str:
    return "Relu"


class Sigmoidal:

  def evaluate(self, val:float)->float:
    e = math.exp(-val);
    return 1 / (1 + e);

  def derivate(self, val:float)->float:
    e = math.exp(val);
    p = math.pow(1 + math.exp(val), 2);
    return (e/p);

  def __str__(self)->str:
    return "Sigmoidal"

class Tahn:

  def evaluate(self, val:float)->float: 
    return math.tanh(val);
   
  def derivate(self, val:float)->float:
    return 1 / math.pow(math.cosh(val), 2);

  def __str__(self)->str:
    return "Tahn"


