import sys
import math
import uuid

from mlp.usable import Util, StringBuffer
from mlp.data   import Example, Value, ValueBuilder

try:
  from ._version import version as __version__
except ImportError:
  __version__ = 'unknown'


class Bias(object):
  neuron   = None
  value    = 0.00
  weight   = 0.00
  momentum = 0.00

  def __init__(self, neuron, value=0.00, weight=0.00, momentum=0.00): 
    self.neuron   = neuron
    self.value    = value
    self.weight   = weight
    self.momentum = momentum
    self.neuron.setBias(self)

  def remove(): 
    self.neuron.setBias(null);

  def reweight(eta, alpha): 
    tmp           = eta * self.value * self.neuron.getError()
    self.weight  += tmp + (alpha * self.momentum)
    self.momentum = tmp

  def getNeuron(self):
    return self.neuron

  def getValue(self):
    return self.value;

  def getWeight(self):
    return self.weight;
  
  def asString(self):
    return "bv: {}, bw: {}, bm: {}".format(self.value, self.weight, self.momentum)
  
  def toString(self):
    return "Bias(v: {} w: {} m: {})".format(self.value, self.weight, self.momentum)

class Neuron(object):
  parent = None 
  bias   = None  # input bias
  inval  = 0.0   # computed input value
  outval = 0.0   # computed output value
  delta  = 0.0   # computed error delta
  error  = 0.0   # computed signal error
  uuid   = None  # UUID identifier for the Neuron
  inputs = []    # list of inputs connections 
  output = []    # list of output connections
 
  def __init__(self, parent, inval=None, outval=None, delta=None, error=None): 
    self.uuid   = uuid.uuid4()
    self.parent = parent
    self.inval  = inval
    self.outval = outval 
    self.delta  = delta
    self.error  = error

  def id(self):
    return str(self.uuid)

  def addOutputConnection(self, conn):
    self.output.append(conn);

  def getOutputConnections(self): 
    return self.output;

  def getOutputValue(self)->float:
    return self.outval if self.outval != None else 0.00
  
  def addInputsConnection(self, conn): 
    self.inputs.append(conn)

  def getInputConnections(self): 
    return self.inputs

  def setBias(self, bias): 
    self.bias = bias

  def getBias(self): 
    return self.bias 

  def getError(self)->float:
    return self.error if self.error != None else 0.0001

  def getNeuronIndex(self): 
    return self.parent.getNeuronIndex(self);
  
  def getLayerIndex(self): 
    return self.parent.getLayerIndex();

  def setOutputValue(self, value):
    self.outval = value.value 

  def calculateOutput(self)->None:
    self._calculateInput()
    if self.parent.isFunctional():
      self.outval = self.parent.evaluate(self.inval * self.parent.parentGain())

  def calculateError(self, target:Value=None)->None:
    if target != None:
      # the difference between our target value and our computed value
      self.delta = target.value - self.getOutputValue() 
    else:
      self.delta = 0.0;
      for i in range(len(self.output)):
        self.delta += (self.output[i].destinationNeuron().getError() * self.output[i].getWeight())
    if self.parent.isFunctional(): 
      self.error = self.delta * self.parent.parentGain() * self.parent.derivate(self.inval)

    if self.parent.isFunctional():
      self.error = self.delta * self.parent.parentGain() * self.parent.derivate(self.inval)
      if self.error > 1.0 : self.error * 0.0001 # WTF???

  def toString(self)->str:
    sb = StringBuffer();
    sb.append("Neuron:{}\n".format(self.uuid))
    sb.append("  Input:  {}\n".format(self.inval))
    sb.append("  Output: {}\n".format(self.outval))
    sb.append("  Error:  {}\n".format(self.error))
    sb.append("  Delta:  {}\n".format(self.delta))
    return sb.toString()

  def __str__(self)->str:
    return self.toString()

  def _calculateInput(self)->None:
    total = 0.0;
    for i in range(len(self.inputs)): 
      c = self.inputs[i]
      total += c.sourceNeuron().getOutputValue() * c.getWeight();
    if self.bias != None:
      total += (self.bias.getValue() * self.bias.getWeight())
    self.inval = total

class Connection(object):
  uuid     = None
  weight   = 0.0;
  momentum = 0.0  
  nsrc     = None # source neuron
  ndst     = None # destination neuron

  def __init__(self, src, dst, weight=Util.randomNetworkWeight()): 
    src.addOutputConnection(self)
    dst.addInputsConnection(self)

    self.uuid   = uuid.uuid4()
    self.nsrc   = src
    self.ndst   = dst
    self.weight = weight

  def reweight(self, eta, alpha):
    tmp            = (eta * self.nsrc.getOutputValue() * self.ndst.getError());
    self.weight   += tmp + (alpha * self.momentum);
    self.momentum  = tmp;

  def getUUID(self):
    return self.uuid

  def sourceNeuron(self):
    return self.nsrc

  def destinationNeuron(self):
    return self.ndst

  def getWeight(self)->float:
    return self.weight if self.weight != None else 0.00

  def getIndex(self):
    return self.nsrc.getNeuronIndex()

  def getLayerIndex(self):
    return self.nsrc.getLayerIndex()

  def toString(self):
    return self.getUUID()


class Layer(object):
  mlp      = None
  uuid     = None
  count    = 0 
  func     = None
  neurons  = []
  form     = "HiddenLayer";

  def __init__(self, mlp, func=None, count=0):
    self.mlp      = mlp;
    self.func     = func;
    self.uuid     = uuid.uuid4()
    self.count    = count;
    self.mlp.addLayer(self);

    for i in range(self.count):
      self.neurons.append(Neuron(self))

  #@XmlAttribute(name="type")
  def getType(self) -> str:
    return self.form

  #@XmlAttribute(name="function") 
  def getFunction(self)->int:
    return -1 if not self.isFunctional() else self.func.function()

  #@XmlAttribute(name="count")
  def getCount()->int:
    """
    Returns the count of neurons in this layer

    Parameters:
    None 

    Returns:
    int            The count of neurons in the layer
    """
    return self.size();

  def size(self)->int:
    return len(self.neurons)

  def isFunctional(self)->bool:
    """
    Is there a function associated with this layer?

    Parameters:
    None

    Returns:
    bool           True if an activation function is assocated with the network  
    """
    return (self.func != None);

  def evaluate(self, val:float)->float:
    return self.func.evaluate(val)

  def derivate(self, val:float)->float: 
    return self.func.derivate(val)

  def parentGain(self)->float:
    return self.mlp.gain

  def getUUID(self)->str:
    return self.uuid

  def getNeuron(self, index:int)->Neuron:
    """
    Returns a Neuron by index number

    Parameters:
    index(int)        The index number corresponding witht he Neuron in this layer

    Returns
    Neuron            The Neuron at postion 'index'  
    """
    if (index >= 0 and index < self.size()):
      return self.neurons.get(index);
    
    return None;

  def addNeuron(self, neuron: Neuron)->None:
    """
    Add a Neuron to the layer

    Parameters:
    Neuron            The Neuron we'd like to add

    Returns:
    None
    """ 
    if (not self.contains(neuron)):
      self.neurons.append(neuron);
      self.count = len(self.neurons)

  #@XmlElement(name="neuron") 
  def getNeurons(self)->list:
    """
    Returns a list of Neurons residing in this layer

    Parameters:
    None

    Returns
    list              A list of Neurons within the layer
    """
    return self.neurons;

  def getNeuronIndex(self, neuron:Neuron)->int:
    """
    Returns the index number of the neuron within this layer

    Parameters:
    Neuron           The Neuron that we want to locate

    Returns:
    int              The Neuron index number, 0 for the first neuron and neurons.size()-1 for the last
    """
    return self_.neurons.index(neuron)

  def getLayerIndex(self)->int:
    """
    Returns the index number of the layout within the MLP

    Parameters:
    None

    Returns:
    int              The layer index, typically 0 for input and layers.size()-1 for output
    """
    return self.mlp.getLayerIndex(self);

  def parentGain(self)->float:
    return self.mlp.gain

  def contains(self, neuron:Neuron)->bool: 
    """
    Determine whether or not a Neuron is in this layer

    Parameters:
    Neuron           A neuron to locate within this layer
  
    Returns:
    bool             True if in the layer, False if not
    """
    if (neuron == None): 
      return False

    return any(x.uuid == neuron.uuid for x in self.neurons)
    return False 

  def MSE(self)->float: # mean square error
    mse = 0.00
    if self.size() == 0: return 0.00
    for i in range(len(self.neurons)):
      mse += math.pow(self.neurons[i].getError(), 2)
    return mse / self.size()

  def MAE(self)->float: # mean abs error
    mae = 0.00
    if self.size() == 0: return 0.00
    for i in range(len(self.neurons)):
      mae += abs(self.neurons[i].getError())
    return mae / self.size()

class InputLayer(Layer):
  form     = "InputLayer";
  neurons  = []

  def __init__(self, mlp, func=None, count=0):
    Layer.__init__(self, mlp, func, count)

  def size(self)->int:
    return 2

  def setInputValues(self, values:list)->bool:
    if len(values) != self.count: return False

    for i in range(len(values)):
      (self.neurons[i]).setOutputValue(values[i])
    return True

  def toString(self)->str:
    sb = StringBuffer();
    sb.append("{} Size: {} ({})\n".format(self.form, self.count, len(self.neurons)))
    for i in range(len(self.neurons)):
      sb.append(str(self.neurons[i]))
    return sb.toString()

  def __str__(self)->str:
    return self.toString()

class OutputLayer(Layer):
  form     = "OutputLayer";
  neurons  = []

  def __init__(self, mlp, func=None, count=0):
    Layer.__init__(self, mlp, func, count)

  def getOutputValues(self)->list:
    out = []
    for i in range(len(self.neurons)):
      out.append(ValueBuilder.asAnalog(self.neurons[i].getOutputValue()))
    return out

  def calculateTargetError(self, target:list)->bool:
    if len(target) != self.count: return False 
    for i in range(len(self.neurons)):
      self.neurons[i].calculateError(target[i]);
    return True;
 
class MLP(object):
  eta       = 0.30;
  alpha     = 0.75;
  gain      = 0.99;
  mae       = 0.02;
  bval      = 1.0;
  layers    = [];
  examples  = [];
  _instance = None

  def __new__(cls, arg, func=None, auto=True):
    # Singleton
    if cls._instance is None:
      cls._instance = super(MLP, cls).__new__(cls)
      if type(arg) == str:
        cls._instance._fromFile(arg)
      elif type(arg) == list:
        cls._instance._fromArgs(arg, func, auto)
      else:
       print("Instantiate from a file or configure with arguments:\n  MLP('filename')\n  MLP([2,2,1], function, True)")
       sys.exit(1)

    return cls._instance

  def addLayer(self, layer): 
    self.layers.append(layer)

  def getLayer(self, index:int)->Layer:
    if index < 0 or index > len(self.layers)-1: return None
    return self.layers[index]    

  def getLayerIndex(self, layer:Layer)->int:
    return self.layers.index(layer)

  def  getInputLayer(self)->InputLayer:
    if len(self.layers) > 0:
      tmp = self.layers[0]
      if isinstance(tmp, InputLayer):
        return tmp;
    return None

  def getOutputLayer(self)->OutputLayer:
    if len(self.layers) > 0:
      tmp = self.layers[len(self.layers)-1]
      if isinstance(tmp, OutputLayer):
        return tmp;
    return None

  def MSE(self)->float:
    return (self.getOutputLayer()).MSE()

  def MAE(self)->float:
    return (self.getOutputLayer()).MAE()

  def isConstructed(self)->bool:
    return (self.getInputLayer() != None and self.getOutputLayer() != None)

  def predict(self, values:list)->list:
    if self._rote(values, None, False) == True:
      return self.getOutputLayer().getOutputValues()
    else:
      return "HA HA!"

  def addExample(self, inputs:list, target:list)->bool: 
    if not self.isConstructed(): return False
    if len(inputs) > (self.getInputLayer()).size():  return False;
    if len(target) > (self.getOutputLayer()).size(): return False;
    self.examples.append(Example(inputs, target))
    return True

  def learnByExample(self, count:int)->bool:
    orig = count
    while count > 0:
      i = Util.randomIndex(len(self.examples))
      if not self.learn((self.examples[i]).getInputs(), (self.examples[i]).getTarget()):
        return False
      if count % 5000 == 0:
        print("Count: {}, MSE: {}, MAE: {}".format(orig-count, self.MSE(), self.MAE()))
      if self.MSE() <= 0.0000000000000001 and self.MAE() < 0.0000000000001:
        print("Done: MSE: {}, MAE: {} (errors under threshold)".format(self.MSE(), self.MAE()))
        return True;
      count -= 1
    print("Done: MSE: {}, MAE: {} (reps complete:{})".format(self.MSE(), self.MAE(), count))
    return True

  def learn(self, inputs:list, target:list)->bool:
    if len(inputs) == (self.getInputLayer()).size():
      return self._rote(inputs, target, True)
    return False 

  def _rote(self, inputs:list, target:list=None, training=False)->bool:
    if not self.isConstructed(): return False
    inner = self.getInputLayer()
    if inner.setInputValues(inputs):
      self._propagateSignal() 

    if target == None: return not training

    outer = self.getOutputLayer()
    if outer.calculateTargetError(target):
      if not training: return True
      return self._backPropagateError()
    return False
  
  def _propagateSignal(self):
    if self.isConstructed(): 
      level = 1
      while level < len(self.layers):
        neurons = self.layers[level].getNeurons()
        for i in range(len(neurons)):
          neurons[i].calculateOutput()
        level += 1
      return True
    return False

  def _backPropagateError(self)->bool:
    if not self.isConstructed(): return False;

    indx = len(self.layers)-1; 
    bias = None
    while indx >= 0:
      axons = self.layers[indx].getNeurons()
      for i in range(len(axons)):
        if indx < len(self.layers)-1:
          if indx > 0: axons[i].calculateError()
          cons = axons[i].getOutputConnections()
          for j in range(len(cons)):
            cons[j].reweight(self.eta, self.alpha)
          bias = axons[i].getBias()
      indx -= 1
    return True

  def _fromFile(self, name):
    print("from file")
    self.eta = .9

  def _fromArgs(self, count:list, func:callable, auto:bool)->object:
    for i in range(len(count)):
      layer = None
      if i == 0:              # Input layer
        layer = InputLayer(self, None, count[i])
      else: 
        if i == len(count)-1: # Output layer
          layer = OutputLayer(self, func, count[i])
        else:                 # Hidden layer(s)
          layer = Layer(self, func, count[i])
        if auto == True:      # Note: we don't autoconnect the input layer
           self._autoConnect(self.getLayer(i-1), layer)
        neurons = layer.getNeurons()
        for i in range(len(neurons)):
          Bias(neurons[i], self.bval)
    #for i in range(len(self.layers)):
    #  ns = self.layers[i].getNeurons()
    #  print(self.layers[i].form)
    #  for j in range(len(ns)):
    #    print(ns[j].uuid)

  def _autoConnect(self, src:Layer, dst:Layer)->None:
    if src.uuid != dst.uuid:
      srcNeurons = src.getNeurons()
      dstNeurons = dst.getNeurons()
      for i in range(len(srcNeurons)):
        for j in range(len(dstNeurons)):
          Connection(srcNeurons[i], dstNeurons[j])


