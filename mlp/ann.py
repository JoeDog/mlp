import uuid

try:
  from ._version import version as __version__
except ImportError:
  __version__ = 'unknown'


class Neuron:
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
  
  def addInputsConnection(self, conn): 
    self.inputs.append(conn)

  def getInputConnections(self): 
    return self.inputs

  def setBias(self, bias): 
    self.bias = bias

  def getBias(self): 
    return self.bias 

  def getNeuronIndex(self): 
    return self.parent.getNeuronIndex(self);
  
  def getLayerIndex(self): 
    return self.parent.getLayerIndex();

  def setOutputValue(value): # XXX
    this.outval = value.getValue() 


